# build_corridor.py
# Units + exits -> free-space mask (near-units & exit-connected) -> skeleton
# -> centerline points -> connectors (constrained to free) -> polylines -> stitching

import argparse, json, os
import numpy as np
import cv2
from pathlib import Path
from dotenv import load_dotenv, find_dotenv  # pip install python-dotenv

load_dotenv(find_dotenv())
OUT_DIR = os.getenv("OUT_DIR") or str(Path(__file__).resolve().parent / "out")

# ---------- utils ----------
def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_outdir(p):
    os.makedirs(p, exist_ok=True)

def data_shape(units, exits):
    xs, ys = [], []
    for u in units:
        for x, y in u.get("polygon_points", []) or []:
            xs.append(float(x)); ys.append(float(y))
        c = u.get("centroid")
        if c:
            xs.append(float(c[0])); ys.append(float(c[1]))
    for e in exits:
        ep = e.get("exit_point")
        if ep:
            xs.append(float(ep[0])); ys.append(float(ep[1]))
    if not xs or not ys:
        raise RuntimeError("Could not derive canvas size from data.")
    W = int(max(xs) + 50); H = int(max(ys) + 50)
    return (H, W)

# ---------- masks ----------
def rasterize_units(units, shape, ds):
    H, W = shape
    m = np.zeros((H//ds, W//ds), np.uint8)
    for u in units:
        pts = u.get("polygon_points") or []
        if len(pts) < 3:
            continue
        arr = np.array([[int(x/ds), int(y/ds)] for x, y in pts], np.int32)
        cv2.fillPoly(m, [arr], 255)  # 255 = obstacle (units)
    return m

def exit_hull(exits, shape, ds):
    """Focus area so skeletonization bahar ki taraf na bhaage."""
    H, W = shape
    pts = [tuple(e["exit_point"]) for e in exits
           if e.get("is_corridor") and e.get("exit_point")]
    m = np.zeros((H//ds, W//ds), np.uint8)
    if len(pts) >= 3:
        arr = np.array([[int(x/ds), int(y/ds)] for x, y in pts], np.int32)
        hull = cv2.convexHull(arr)
        cv2.fillConvexPoly(m, hull, 255)  # 255 = focus
    else:
        m[:] = 255
    return m

def _keep_components_touching_exits(free, exits, ds, exit_radius_px=12, min_comp_area_px=800):
    """Sirf un FREE components ko rakho jo kisi exit ko touch karte ho."""
    if free.max() == 0:
        return free
    h, w = free.shape
    # small disk of exits
    er = max(1, int(round(exit_radius_px/ds)))
    em = np.zeros_like(free, np.uint8)
    for e in exits:
        if not (e.get("is_corridor") and e.get("exit_point")):
            continue
        x, y = e["exit_point"]
        cx, cy = int(round(x/ds)), int(round(y/ds))
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(em, (cx, cy), er, 255, -1, cv2.LINE_AA)

    # connected components on FREE
    num, labels = cv2.connectedComponents((free > 0).astype(np.uint8))
    if num <= 1:
        return free

    # area threshold in downscaled pixels
    min_area_ds = max(1, int(round(min_comp_area_px/(ds*ds))))
    areas = np.bincount(labels.ravel())

    keep = np.zeros_like(free, np.uint8)
    for lab in range(1, num):
        if areas[lab] < min_area_ds:
            continue
        comp_mask = (labels == lab)
        # touch exits?
        if (comp_mask & (em > 0)).any():
            keep[comp_mask] = 255

    return keep

def make_free_mask(units, exits, shape, ds,
                   clearance_px=8, near_px=200,
                   exit_radius_px=12, min_comp_area_px=800):
    """
    Final FREE mask (255=free) =
      (inside exit hull) ∧ (NOT dilated units) ∧ (within near_px of any unit)
      filtered to components that touch an exit.
    """
    um = rasterize_units(units, shape, ds)     # 255 = obstacles
    hm = exit_hull(exits, shape, ds)           # 255 = focus

    # clearance around walls/shops
    k = max(1, int(round(clearance_px/ds)))
    k_ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
    grown = cv2.dilate(um, k_ell, iterations=1)

    # near-units band (white outside cut)
    inv = cv2.bitwise_not(um)  # 255 = free/background
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 5)  # distance to nearest UNIT in downscaled px
    near_r = max(1, int(round(near_px/ds)))
    near = (dt <= float(near_r)).astype(np.uint8) * 255

    free0 = cv2.bitwise_and(hm, cv2.bitwise_not(grown))     # drop units (+clearance)
    free0 = cv2.bitwise_and(free0, near)                    # keep only near-units band

    # smooth
    free0 = cv2.medianBlur(free0, 3)

    # keep only exit-connected components (removes outer white)
    free = _keep_components_touching_exits(free0, exits, ds,
                                           exit_radius_px=exit_radius_px,
                                           min_comp_area_px=min_comp_area_px)

    return free

# ---------- skeleton ----------
def skeletonize(bin_img):
    img  = (bin_img > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    for _ in range(5000):  # safe cap
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp   = cv2.subtract(img, opened)
        skel   = cv2.bitwise_or(skel, temp)
        img    = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel

def skeleton_points(skel, ds):
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return np.zeros((0,2), dtype=float)
    return np.stack([(xs+0.5)*ds, (ys+0.5)*ds], axis=1).astype(float)

# ---------- connectors ----------
def bresenham_ds(p_full, q_full, ds):
    x0, y0 = int(round(p_full[0]/ds)), int(round(p_full[1]/ds))
    x1, y1 = int(round(q_full[0]/ds)), int(round(q_full[1]/ds))
    pts=[]; dx=abs(x1-x0); dy=-abs(y1-y0)
    sx=1 if x0<x1 else -1; sy=1 if y0<y1 else -1
    err=dx+dy; x,y=x0,y0
    while True:
        pts.append((x,y))
        if x==x1 and y==y1: break
        e2=2*err
        if e2>=dy: err+=dy; x+=sx
        if e2<=dx: err+=dx; y+=sy
    return pts

def valid_line_in_free(p, q, free_mask, ds):
    h, w = free_mask.shape
    for (x,y) in bresenham_ds(p,q,ds):
        if not (0<=x<w and 0<=y<h): return False
        if free_mask[y,x] == 0:     return False
    return True

def connect_exits_to_skeleton(center_pts, exits, free_mask, ds):
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(center_pts) if len(center_pts) else None
    except Exception:
        tree = None

    q_exits = [e for e in exits if e.get("is_corridor") and e.get("exit_point")]
    connectors = []
    if len(center_pts) == 0:
        return connectors

    for e in q_exits:
        p = np.array(e["exit_point"], dtype=float)
        if tree is not None:
            _, idx = tree.query(p, k=1)
            q = center_pts[int(idx)]
            if valid_line_in_free(p, q, free_mask, ds):
                connectors.append({
                    "block": e.get("block"),
                    "unit": e.get("unit"),
                    "from_exit": p.tolist(),
                    "to_corridor": q.tolist(),
                    "polyline": [p.tolist(), q.tolist()]
                })
        else:
            d = np.hypot(center_pts[:,0]-p[0], center_pts[:,1]-p[1])
            order = np.argsort(d)[:1000]
            for j in order:
                q = center_pts[j]
                if valid_line_in_free(p, q, free_mask, ds):
                    connectors.append({
                        "block": e.get("block"),
                        "unit": e.get("unit"),
                        "from_exit": p.tolist(),
                        "to_corridor": q.tolist(),
                        "polyline": [p.tolist(), q.tolist()]
                    })
                    break
    return connectors

# ---------- dots -> polylines ----------
def centerpoints_to_polylines(center_pts, ds):
    grid = np.c_[np.round(center_pts[:,0]/ds - 0.5).astype(int),
                 np.round(center_pts[:,1]/ds - 0.5).astype(int)]
    pixels = set(map(tuple, grid))
    if not pixels:
        return []

    N8 = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
    def nbrs(p):
        x,y = p
        for dx,dy in N8:
            q = (x+dx, y+dy)
            if q in pixels:
                yield q

    polylines = []
    visited_pix = set()

    def walk(comp_set, deg_map, a, b, used_edges):
        path = [a, b]
        prev, cur = a, b
        used_edges.add((a,b)); used_edges.add((b,a))
        guard = 0
        while deg_map.get(cur, 0) == 2:
            guard += 1
            if guard > 200000:
                break
            nxts = [n for n in nbrs(cur) if n != prev and n in comp_set]
            if not nxts:
                break
            nxt = nxts[0]
            if (cur, nxt) in used_edges:
                break
            used_edges.add((cur, nxt)); used_edges.add((nxt, cur))
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    for start in list(pixels):
        if start in visited_pix:
            continue
        comp = []
        stack = [start]
        visited_pix.add(start)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in nbrs(u):
                if v not in visited_pix:
                    visited_pix.add(v)
                    stack.append(v)

        comp_set = set(comp)
        deg_map = {p: sum((n in comp_set) for n in nbrs(p)) for p in comp}
        endpoints = [p for p in comp if deg_map[p] != 2]
        used_edges = set()

        if endpoints:
            for a in endpoints:
                for b in nbrs(a):
                    if b not in comp_set or (a, b) in used_edges:
                        continue
                    seg = walk(comp_set, deg_map, a, b, used_edges)
                    if len(seg) >= 2:
                        polylines.append(seg)
        else:  # cycle
            a = comp[0]
            nb = [n for n in nbrs(a) if n in comp_set]
            if nb:
                seg = walk(comp_set, deg_map, a, nb[0], used_edges=set())
                if len(seg) >= 3:
                    polylines.append(seg)

    def pix_to_full(poly):
        return [[(x+0.5)*ds, (y+0.5)*ds] for (x,y) in poly]
    return [pix_to_full(pl) for pl in polylines]

# ---------- stitching ----------
def _angle_between(u, v):
    if np.linalg.norm(u) < 1e-6 or np.linalg.norm(v) < 1e-6:
        return 0.0
    uu = u / (np.linalg.norm(u)+1e-12)
    vv = v / (np.linalg.norm(v)+1e-12)
    c = np.clip(np.dot(uu, vv), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def _nearest_endpoints(polys):
    ends = []  # (x,y, poly_idx, which_end, dir_vec)
    for i, pl in enumerate(polys):
        if len(pl) < 2:
            continue
        p0, p1 = np.array(pl[0], float), np.array(pl[1], float)
        q0, q1 = np.array(pl[-1], float), np.array(pl[-2], float)
        d_start = p1 - p0
        d_end   = q0 - q1
        ends.append((p0[0], p0[1], i, 0, d_start))
        ends.append((q0[0], q0[1], i, 1, d_end))
    pts = np.array([[e[0], e[1]] for e in ends], dtype=np.float32)
    return ends, pts

def _bres_ok(p, q, free_mask, ds):
    p = tuple(map(float, p)); q = tuple(map(float, q))
    return valid_line_in_free(p, q, free_mask, ds)

def stitch_polylines_on_free(polylines, free_mask, ds,
                             max_join_dist=15, angle_thresh_deg=25,
                             max_iters=4):
    if not polylines:
        return polylines
    polys = [list(pl) for pl in polylines]

    try:
        from scipy.spatial import cKDTree
        use_tree = True
    except Exception:
        use_tree = False

    for _ in range(max_iters):
        ends, pts = _nearest_endpoints(polys)
        if len(pts) == 0:
            break
        if use_tree:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
        changed = False
        used = set()

        for i in range(len(ends)):
            if i in used: 
                continue
            x, y, ip, ie, dvec_i = ends[i]
            if len(polys[ip]) < 2:
                continue

            if use_tree:
                idxs = tree.query_ball_point([x, y], r=max_join_dist)
            else:
                d = np.hypot(pts[:,0]-x, pts[:,1]-y)
                idxs = list(np.where(d <= max_join_dist)[0])

            idxs = sorted((j for j in idxs if j != i), key=lambda j: (pts[j,0]-x)**2 + (pts[j,1]-y)**2)
            for j in idxs:
                if j in used: 
                    continue
                X, Y, jp, je, dvec_j = ends[j]
                if ip == jp:
                    continue

                v = np.array([X-x, Y-y], float)
                ang_i = min(_angle_between(dvec_i,  v), _angle_between(-dvec_i, v))
                ang_j = min(_angle_between(dvec_j, -v), _angle_between(-dvec_j, -v))
                if ang_i > angle_thresh_deg or ang_j > angle_thresh_deg:
                    continue

                if not _bres_ok((x,y), (X,Y), free_mask, ds):
                    continue

                A = polys[ip]; B = polys[jp]
                if ie == 1 and je == 0:     new = A + B
                elif ie == 0 and je == 1:   new = B + A
                elif ie == 1 and je == 1:   new = A + B[::-1]
                else:                        new = B[::-1] + A

                polys[ip] = new
                polys[jp] = []
                used.add(i); used.add(j)
                changed = True
                break

        polys = [pl for pl in polys if len(pl) >= 2]
        if not changed:
            break
    return polys

# ---------- simplify ----------
def simplify_polylines(polylines, eps=0.0):
    if eps <= 0:
        return polylines
    out = []
    for pl in polylines:
        arr = np.array(pl, np.float32)
        if len(arr) < 3:
            out.append(pl); continue
        approx = cv2.approxPolyDP(arr.astype(np.float32), eps, False)
        out.append([[float(x), float(y)] for [x,y] in approx.reshape(-1,2)])
    return out

# ---------- overlay ----------
def draw_overlay(H, W, polylines, connectors, out_png):
    canvas = np.ones((H, W, 3), np.uint8)*255
    for pl in polylines:
        arr = np.array(pl, np.int32)
        if len(arr)>=2:
            cv2.polylines(canvas, [arr], False, (200,0,200), 2, cv2.LINE_AA)
    for c in connectors:
        a = tuple(map(int, c["from_exit"]))
        b = tuple(map(int, c["to_corridor"]))
        cv2.line(canvas, a, b, (0,128,0), 2, cv2.LINE_AA)
        cv2.circle(canvas, a, 3, (0,0,255), -1, cv2.LINE_AA)
    cv2.imwrite(out_png, canvas)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Build corridor network from unit polygons + exit points (color-independent).")
    ap.add_argument("--units", default=os.path.join(OUT_DIR, "points_with_centroid.json"),
                    help="units json (default: OUT_DIR/points_with_centroid.json)")
    ap.add_argument("--exits", default=os.path.join(OUT_DIR, "centroid_to_corridor_exit.json"),
                    help="exits json (default: OUT_DIR/centroid_to_corridor_exit.json)")
    ap.add_argument("--outdir", default=OUT_DIR,
                    help="output directory (default: OUT_DIR from .env or ./out)")
    ap.add_argument("--downscale", type=int, default=3, help="raster downscale factor (2–5 typical)")
    ap.add_argument("--clearance", type=int, default=8, help="wall clearance in original pixels")

    # NEW: corridor locality + exit connectivity
    ap.add_argument("--near-px", type=int, default=200, help="keep FREE only within this many px of any unit (cuts outside white)")
    ap.add_argument("--exit-radius", type=int, default=12, help="exit touch radius for component keep (orig px)")
    ap.add_argument("--min-comp-area", type=int, default=800, help="drop tiny FREE components (orig px^2)")

    # gap handling
    ap.add_argument("--close-px",  type=int, default=4, help="close tiny gaps in FREE before skeleton (orig px)")
    ap.add_argument("--bridge-px", type=int, default=8, help="bridge tiny breaks on SKELETON then re-thin (orig px)")

    # stitching
    ap.add_argument("--stitch-dist",  type=int, default=18, help="max distance to join polyline endpoints (orig px)")
    ap.add_argument("--stitch-angle", type=int, default=30, help="max angle diff (deg) for joining")
    ap.add_argument("--stitch-iters", type=int, default=4, help="stitch iterations")

    # simplify
    ap.add_argument("--simplify-eps", type=float, default=1.2, help="Douglas-Peucker epsilon (orig px), 0=off")
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    if not os.path.isfile(args.units):  raise FileNotFoundError(args.units)
    if not os.path.isfile(args.exits):  raise FileNotFoundError(args.exits)
    units = load_json(args.units)
    exits = load_json(args.exits)

    H, W = data_shape(units, exits)

    # 1) Free mask (+ optional close)
    free = make_free_mask(
        units, exits, (H, W), args.downscale,
        clearance_px=args.clearance,
        near_px=args.near_px,
        exit_radius_px=args.exit_radius,
        min_comp_area_px=args.min_comp_area
    )
    free_for_skel = free.copy()
    if args.close_px > 0:
        k = max(1, int(round(args.close_px / args.downscale)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
        free_for_skel = cv2.morphologyEx(free, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(os.path.join(args.outdir, "debug_free_mask.png"), free_for_skel)
    if int(cv2.countNonZero(free_for_skel)) == 0:
        raise RuntimeError("Free mask empty. Check inputs/params.")

    # 2) Skeleton (raw)
    skel = skeletonize(free_for_skel)
    cv2.imwrite(os.path.join(args.outdir, "debug_skeleton_raw.png"), skel)

    # 3) Bridge tiny breaks on skeleton + re-thin
    if args.bridge_px > 0:
        kb = max(1, int(round(args.bridge_px / args.downscale)))
        kernel_b = cv2.getStructuringElement(cv2.MORPH_RECT, (2*kb+1, 2*kb+1))
        skel = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, kernel_b)
        skel = skeletonize(skel)
    cv2.imwrite(os.path.join(args.outdir, "debug_skeleton_bridged.png"), skel)

    # 4) Centerline points
    center_pts = skeleton_points(skel, args.downscale)
    json.dump({"points": center_pts.tolist(), "downscale": args.downscale},
              open(os.path.join(args.outdir, "corridor_centerlines.json"), "w"), indent=2)

    # 5) Exit→corridor connectors (check against FREE mask)
    connectors = connect_exits_to_skeleton(center_pts, exits, free_for_skel, args.downscale)
    json.dump({"connectors": connectors},
              open(os.path.join(args.outdir, "corridor_connectors.json"), "w"), indent=2)

    # 6) Dots→polylines
    polylines = centerpoints_to_polylines(center_pts, args.downscale)

    # 7) Smart stitching to remove gaps (distance + angle + free check)
    polylines = stitch_polylines_on_free(polylines, free_for_skel, args.downscale,
                                         max_join_dist=args.stitch_dist,
                                         angle_thresh_deg=args.stitch_angle,
                                         max_iters=args.stitch_iters)

    # 8) Optional simplify
    polylines = simplify_polylines(polylines, eps=float(args.simplify_eps))

    # Save polylines + network
    json.dump({"polylines": polylines, "downscale": args.downscale},
              open(os.path.join(args.outdir, "corridor_polylines.json"), "w"), indent=2)

    network = {
        "centerline_points": center_pts.tolist(),
        "connectors": connectors,
        "polylines": polylines,
        "meta": {
            "downscale": args.downscale,
            "wall_clearance_px": args.clearance,
            "near_px": args.near_px,
            "exit_radius_px": args.exit_radius,
            "min_comp_area_px": args.min_comp_area,
            "close_px": args.close_px,
            "bridge_px": args.bridge_px,
            "stitch_dist": args.stitch_dist,
            "stitch_angle": args.stitch_angle,
            "stitch_iters": args.stitch_iters,
            "simplify_eps": args.simplify_eps,
        }
    }
    json.dump(network, open(os.path.join(args.outdir, "corridor_network.json"), "w"), indent=2)

    # 9) Overlay (debug)
    draw_overlay(H, W, polylines, connectors, os.path.join(args.outdir, "corridor_lines_overlay.png"))

    print(f"[OK] centerline points: {len(center_pts)}")
    print(f"[OK] connectors:        {len(connectors)}")
    print(f"[OK] polylines:         {len(polylines)}")
    print("Saved in:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
