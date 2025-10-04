# postman.py
# Single-file Chinese Postman: ONE continuous route over corridor polylines.
# Steps:
#   1) Build MultiGraph from polylines (with node snapping)
#   2) Compress graph: contract degree-2 chains into long edges (huge speedup)
#   3) If needed, connect components (configurable) with shortest links OR skip
#   4) Chinese Postman (odd nodes matching + Euler circuit)
#   5) Expand to one continuous polyline + overlay
#
# Usage:
#   python -m pip install networkx scipy numpy opencv-python
#   python postman.py --polylines ".\out\corridor_polylines.json" --snap 8 --compress-iters 6 --connect shortest --outdir ".\out"
#   # Optional: follow original geometry instead of straight lines:
#   python postman.py --style geom --polylines ".\out\corridor_polylines.json" --outdir ".\out"
#
# New/changed options:
#   --style straight|geom     : final drawn route style (default: straight)
#   --connect shortest|none   : connect disconnected components (default shortest) OR skip (use largest component only)
#   --prune-spurs <px>        : remove dead-end edges shorter than this (px) before compression (0 disables)
#   --simplify-eps <px>       : simplify final polyline visually (RDP epsilon in px; 0 disables)
#
import os, json, math, argparse
import numpy as np
import cv2
import networkx as nx

# ------------------- helpers -------------------
def pair_segments(poly):
    return [[poly[i], poly[i+1]] for i in range(len(poly)-1)]

def length_of_coords(coords):
    if not coords or len(coords) < 2:
        return 0.0
    a = np.asarray(coords, dtype=float)
    return float(np.linalg.norm(a[1:] - a[:-1], axis=1).sum())

# --------- endpoint snapping (cluster) ---------
def cluster_points(points, tol):
    """Cluster endpoints within 'tol' px. Uses SciPy KDTree if available, else grid fallback."""
    if len(points) == 0:
        return np.zeros((0,2), float), np.zeros((0,), int)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        parent = list(range(len(points)))
        rank   = [0]*len(points)
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a,b):
            ra, rb = find(a), find(b)
            if ra == rb: return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
        for i, nbrs in enumerate(tree.query_ball_point(points, r=tol)):
            for j in nbrs: union(i, j)
        groups = {}
        for i in range(len(points)):
            r = find(i)
            groups.setdefault(r, []).append(i)
        centers=[]; labels=np.zeros(len(points), int)
        for new_id, idxs in enumerate(groups.values()):
            centers.append(points[idxs].mean(axis=0))
            for k in idxs: labels[k] = new_id
        return np.asarray(centers,float), labels
    except Exception:
        q = np.round(points / max(tol,1e-6)) * tol
        uniq, labels = np.unique(q, axis=0, return_inverse=True)
        return uniq.astype(float), labels

# ---------------- graph building ----------------
def build_graph_from_polylines(polylines, snap_tol=8.0):
    """
    Build an undirected MultiGraph from polylines.
    - Every consecutive pair -> edge with 'length' + 'geom' (list of coords)
    - All vertices snapped within 'snap_tol' so intersections meet.
    """
    endpoints=[]; segs=[]
    for pl in polylines:
        if not pl or len(pl) < 2:
            continue
        for a, b in pair_segments(pl):
            a=(float(a[0]), float(a[1])); b=(float(b[0]), float(b[1]))
            endpoints += [a, b]
            segs.append((a, b))
    if not segs:
        return nx.MultiGraph()

    pts = np.asarray(endpoints, float)
    centers, labels = cluster_points(pts, snap_tol)

    G = nx.MultiGraph()
    for i, c in enumerate(centers):
        G.add_node(i, x=float(c[0]), y=float(c[1]))

    k = 0
    for (a, b) in segs:
        u, v = int(labels[k]), int(labels[k+1]); k += 2
        G.add_edge(u, v, length=math.hypot(b[0]-a[0], b[1]-a[1]), geom=[list(a), list(b)])
    return G

# ---------------- graph compression ----------------
def _orient_geom_for_endpoints(geom, ux, uy, vx, vy):
    """Orient a polyline geometry so it starts near (ux,uy) and ends near (vx,vy)."""
    g = geom
    if not g or len(g) < 2:
        return g
    d_start = abs(g[0][0]-ux) + abs(g[0][1]-uy) + abs(g[-1][0]-vx) + abs(g[-1][1]-vy)
    d_rev   = abs(g[-1][0]-ux) + abs(g[-1][1]-uy) + abs(g[0][0]-vx) + abs(g[0][1]-vy)
    return g if d_start <= d_rev else g[::-1]

def compress_degree2_chains(G: nx.MultiGraph, max_iters=3) -> nx.MultiGraph:
    """
    Contract linear chains: if a node has degree==2, merge its two incident edges into one long edge.
    Repeats up to max_iters.
    """
    H = G.copy()
    for _ in range(max_iters):
        to_process = [n for n in list(H.nodes) if H.degree(n) == 2]
        if not to_process:
            break
        changed = False
        for n in to_process:
            if n not in H or H.degree(n) != 2:
                continue
            eds = list(H.edges(n, keys=True, data=True))
            if len(eds) != 2:
                continue
            (u1, v1, k1, d1), (u2, v2, k2, d2) = eds
            a = v1 if u1 == n else u1
            b = v2 if u2 == n else u2
            if a == b and a == n:
                continue
            ax, ay = H.nodes[a]["x"], H.nodes[a]["y"]
            bx, by = H.nodes[b]["x"], H.nodes[b]["y"]
            nx_, ny_ = H.nodes[n]["x"], H.nodes[n]["y"]

            g1 = _orient_geom_for_endpoints(d1.get("geom", []), ax, ay, nx_, ny_)
            g2 = _orient_geom_for_endpoints(d2.get("geom", []), nx_, ny_, bx, by)

            merged = (g1 if g1 else [[ax, ay], [nx_, ny_]]) + (g2[1:] if (g2 and g1) else g2)
            L = length_of_coords(merged)

            try:
                H.remove_edge(u1, v1, k1); H.remove_edge(u2, v2, k2)
            except Exception:
                pass
            if n in H and H.degree(n) == 0:
                try: H.remove_node(n)
                except Exception: pass
            H.add_edge(a, b, length=float(L), geom=merged)
            changed = True
        if not changed:
            break
    return H

def prune_short_spurs(G: nx.MultiGraph, min_len=8.0, max_iters=3) -> nx.MultiGraph:
    """Remove short dead-end edges (< min_len). Iterative so cascading stubs bhi hat jayein."""
    H = G.copy()
    for _ in range(max_iters):
        to_del = []
        for n in list(H.nodes):
            if n not in H:
                continue
            if H.degree(n) == 1:
                eds = list(H.edges(n, keys=True, data=True))
                if not eds:
                    continue
                (u, v, k, d) = eds[0]
                L = float(d.get("length", 0.0))
                if L <= float(min_len):
                    to_del.append((u, v, k))
        if not to_del:
            break
        for u, v, k in to_del:
            if H.has_edge(u, v, k):
                H.remove_edge(u, v, k)
        # remove isolated nodes
        iso = [x for x in list(H.nodes) if H.degree(x) == 0]
        if iso:
            H.remove_nodes_from(iso)
    return H

def connect_components_if_needed(G):
    """If multiple components, connect by shortest straight link between component nodes."""
    comps = list(nx.connected_components(G))
    if len(comps) <= 1:
        return
    base = comps[0]
    for comp in comps[1:]:
        best = None
        for u in base:
            xu, yu = G.nodes[u]["x"], G.nodes[u]["y"]
            for v in comp:
                xv, yv = G.nodes[v]["x"], G.nodes[v]["y"]
                d = math.hypot(xv - xu, yv - yu)
                if best is None or d < best[0]:
                    best = (d, u, v)
        d, u, v = best
        G.add_edge(u, v, length=float(d),
                   geom=[[G.nodes[u]["x"], G.nodes[u]["y"]],
                         [G.nodes[v]["x"], G.nodes[v]["y"]]])
        base = base.union(comp)

# --------------- chinese postman ----------------
def all_pairs_shortest(G, nodes):
    """Shortest path length + node list for each unordered odd-node pair."""
    out = {}
    for s in nodes:
        dist, path = nx.single_source_dijkstra(G, s, weight="length")
        for t in nodes:
            if t <= s:
                continue
            if t in dist:
                out[(s, t)] = (float(dist[t]), path[t])
    return out

def chinese_postman_route(G_in, connect_mode="shortest"):
    """Return (Eulerian-augmented graph, euler_edges_with_keys, total_length)."""
    G = G_in.copy()
    if not nx.is_connected(G):
        if connect_mode == "shortest":
            connect_components_if_needed(G)
        else:
            comps = list(nx.connected_components(G))
            if not comps:
                return G, [], 0.0
            biggest = max(comps, key=len)
            G = G.subgraph(biggest).copy()
            print(f"[i] Not connecting components; using largest component with {len(biggest)} nodes.")

    odd = [n for n in G.nodes if G.degree(n) % 2 == 1]
    if odd:
        sp = all_pairs_shortest(G, odd)
        if not sp:
            return G, [], 0.0
        K = nx.Graph()
        for (u, v), (w, _) in sp.items():
            K.add_edge(u, v, weight=w)
        # Minimum-weight perfect matching on odd nodes
        M = nx.algorithms.matching.min_weight_matching(K, weight="weight")
        # duplicate shortest paths for matched pairs
        for pair in M:
            u, v = tuple(pair)
            if u > v: u, v = v, u
            _, path_nodes = sp[(u, v)]
            for a, b in zip(path_nodes[:-1], path_nodes[1:]):
                data = G.get_edge_data(a, b)
                if data is None:
                    continue
                if isinstance(G, nx.MultiGraph):
                    d = min(data.values(), key=lambda dd: dd.get("length", 1.0))
                else:
                    d = data
                G.add_edge(a, b, **d)

    # Euler edges WITH KEYS so duplicates are preserved
    try:
        if isinstance(G, nx.MultiGraph):
            eul = list(nx.eulerian_circuit(G, keys=True))  # (u,v,k)
        else:
            eul_plain = list(nx.eulerian_circuit(G))       # (u,v)
            eul = [(u, v, 0) for (u, v) in eul_plain]
    except nx.NetworkXError:
        return G, [], 0.0

    if not eul:
        return G, [], 0.0

    # Accurate total length: sum EXACT traversed edge lengths using key
    total_len = 0.0
    for (u, v, k) in eul:
        data = G.get_edge_data(u, v)
        if data is None:
            continue
        d = data[k] if isinstance(G, nx.MultiGraph) else data
        total_len += float(d.get("length", 0.0))

    return G, eul, total_len

def edges_to_coords(G, euler_edges_with_keys, straight=False):
    """Expand Euler edges (with keys) into one polyline; preserves duplicates (full coverage)."""
    coords = []
    for (u, v, k) in euler_edges_with_keys:
        if straight:
            ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
            vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
            part = [[ux, uy], [vx, vy]]
        else:
            data = G.get_edge_data(u, v)
            if data is None:
                continue
            d = data[k] if isinstance(G, nx.MultiGraph) else data
            seg = d["geom"]
            ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
            # orient seg u->v
            if abs(seg[0][0]-ux) + abs(seg[0][1]-uy) <= abs(seg[-1][0]-ux) + abs(seg[-1][1]-uy):
                part = seg
            else:
                part = seg[::-1]

        if not coords:
            coords.extend(part)
        else:
            if coords[-1] == part[0]:
                coords.extend(part[1:])
            else:
                coords.extend(part)
    return coords

def simplify_polyline(points, eps=0.0):
    """Ramer–Douglas–Peucker via OpenCV (visual only). eps=0 means no-op."""
    if not points or eps <= 0:
        return points
    try:
        arr = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(arr, epsilon=float(eps), closed=False)
        return approx.reshape(-1, 2).astype(float).tolist()
    except Exception:
        # tiny fallback: keep every Nth point ~ eps
        step = max(2, int(eps))
        return [points[0]] + points[1:-1:step] + [points[-1]]

# -------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser(description="Chinese Postman (single continuous route) over corridor polylines.")
    ap.add_argument("--polylines", default="out/corridor_polylines.json", help="path to corridor_polylines.json")
    ap.add_argument("--snap", type=float, default=12.0, help="node snap tolerance (px)")
    ap.add_argument("--compress-iters", type=int, default=6, help="degree-2 chain compression passes")
    ap.add_argument("--outdir", default="out", help="output folder")
    ap.add_argument("--overlay-img", default="", help="optional background image")
    ap.add_argument("--overlay-size-from", choices=["data","image"], default="data",
                    help="canvas from route (data) or size of overlay image (image)")
    # CHANGED: default straight lines
    ap.add_argument("--style", choices=["straight","geom"], default="straight",
                    help="Draw style for final route: 'straight' (default) or 'geom' (follow original geometry).")
    ap.add_argument("--connect", choices=["shortest","none"], default="shortest",
                    help="Handle disconnected graphs: shortest to connect components, or none to use only the largest component.")
    ap.add_argument("--prune-spurs", type=float, default=0.0,
                    help="Prune dead-end edges shorter than this length (px). 0 to disable.")
    ap.add_argument("--simplify-eps", type=float, default=0.0,
                    help="Simplify final drawn polyline with approxPolyDP epsilon (px). 0 to disable.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # load polylines
    with open(args.polylines, "r", encoding="utf-8") as f:
        d = json.load(f)
    polylines = d.get("polylines", [])
    if not polylines:
        raise SystemExit("No polylines found. Run build_corridor.py first.")

    # Build graph
    print("[i] Building graph...")
    G_raw = build_graph_from_polylines(polylines, snap_tol=args.snap)
    print(f"[i] Raw graph: nodes={G_raw.number_of_nodes()}, edges={G_raw.number_of_edges()}")

    # Optional spur pruning
    if args.prune_spurs > 0:
        print(f"[i] Pruning short spurs < {args.prune_spurs}px ...")
        G_raw = prune_short_spurs(G_raw, min_len=args.prune_spurs, max_iters=3)
        print(f"[i] After spur prune: nodes={G_raw.number_of_nodes()}, edges={G_raw.number_of_edges()}")

    # Compress degree-2 chains
    print("[i] Compressing degree-2 chains...")
    G = compress_degree2_chains(G_raw, max_iters=args.compress_iters)
    print(f"[i] Compressed graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    # Chinese Postman
    print("[i] Solving Chinese Postman...")
    G_aug, euler_edges, total_len = chinese_postman_route(G, connect_mode=args.connect)

    if not euler_edges:
        print("[warn] No Eulerian route found (graph might be empty or disconnected with no edges).")
        line = []
    else:
        straight = (args.style == "straight")
        line = edges_to_coords(G_aug, euler_edges, straight=straight)

    # Optional visual simplification
    if args.simplify_eps > 0 and len(line) >= 2:
        line = simplify_polyline(line, eps=args.simplify_eps)

    # Save JSON (even if empty)
    out_json = os.path.join(args.outdir, "postman_route.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"line": line, "length": float(total_len)}, f, indent=2)

    # Overlay
    if args.overlay_size_from == "image" and os.path.isfile(args.overlay_img):
        canvas = cv2.imread(args.overlay_img)
        H, W = canvas.shape[:2]
    elif line:
        W = int(max(p[0] for p in line) + 50)
        H = int(max(p[1] for p in line) + 50)
        canvas = np.ones((H, W, 3), np.uint8) * 255
    else:
        H, W = 512, 512
        canvas = np.ones((H, W, 3), np.uint8) * 255

    # Draw
    if len(line) >= 2:
        pts = np.array(line, np.int32)
        cv2.polylines(canvas, [pts], False, (200, 0, 200), 3, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "No route available", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2, cv2.LINE_AA)

    out_png = os.path.join(args.outdir, "postman_route_overlay.png")
    cv2.imwrite(out_png, canvas)

    print("[OK] Postman total length:", round(float(total_len), 2))
    print("[Saved]", out_json)
    print("[Saved]", out_png)

if __name__ == "__main__":
    main()
