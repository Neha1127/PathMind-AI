#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Layout Extractor — auto-select color segmentation vs. structure-based (UPDATED v5)

What's new in v5
----------------
- COLOR mode fix: orange & white masks are processed via **separate contour passes**
  (no OR-union). This prevents the "one giant contour" issue and yields multiple units.
- Strict auto: if orange_frac >= 2%, auto ALWAYS selects COLOR (no fallback).
- Keeps earlier safeguards:
  - COLOR considered meaningful when orange>=1%.
  - White-unit merge is GATED (we do NOT union; we just process separately).
- Optional exports: --export_debug_points, --export_points_json
- NEW: You can now override HSV color ranges from CLI using **--color_preset** or **--hsv_range**.
  Examples:
    --color_preset blue
    --hsv_range 100,50,50:130,255,255   (repeatable)
"""

import os
import json
import argparse
from typing import Tuple, Dict, Any, List

import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv


# --- Optional SciPy imports (snapping); auto-skip if not available ---
try:
    from scipy.spatial import ConvexHull, cKDTree  # type: ignore
    SCIPY_OK = True
except Exception:
    ConvexHull = None  # type: ignore
    cKDTree = None     # type: ignore
    SCIPY_OK = False

# ===================== Tunables / Defaults =====================
# COLOR mode tunables
EPS_ARC_FRAC       = 0.012   # polygon approximation epsilon fraction of perimeter
ERODE_K0           = 2       # erosion kernel base (auto scales by image size)
MIN_UNIT_AREA0     = 220     # small-unit base area (auto scales by image size)

# HSV ranges for peach/orange (OpenCV H: 0..179)---default; can be overridden by CLI
# Each range is a tuple of (lower_bound, upper_bound) in HSV space.
HSV_RANGES = [
    ((  4,  40, 140), ( 22, 255, 255)),  # light peach/orange
    (( 22,  30, 170), ( 32, 255, 255)),  # warm light tones
]

# Quick color presets (OpenCV HSV ranges, H: 0–179). Used by --color_preset
COLOR_PRESETS: Dict[str, List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]] = {
    "orange": [((4, 40, 140), (22, 255, 255)), ((22, 30, 170), (32, 255, 255))],
    "blue":   [((100, 50, 50), (130, 255, 255))],
    "pink":   [((140, 50, 50), (170, 255, 255))],  # magenta/pink
    "red":    [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (179, 255, 255))],
    "yellow": [((20, 70, 70), (35, 255, 255))],
    "green":  [((40, 40, 40), (85, 255, 255))],
    "cyan":   [((85, 50, 50), (100, 255, 255))],
    "purple": [((125, 50, 50), (150, 255, 255))],
}

# white-ish unit detection (COLOR mode; background removed)
WHITE_S_MAX = 40
WHITE_V_MIN = 230

# STRUCTURE mode tunables (fractions of total image area)
BLOCK_AREA_FRAC    = 0.0010
UNIT_AREA_FRAC     = 0.00005
ADAPT_BLOCK_SIZE   = 21
ADAPT_C            = 10

# Auto-mode heuristics
COLOR_PIXEL_FRAC_LO = 0.01    # 1%: color plausible
COLOR_PIXEL_FRAC_HI = 0.40
MIN_UNITS_OK        = 3
WHITE_IN_COLOR_MIN_ORANGE_FRAC = 0.01  # allow white processing only if orange>=1%

# STRICT rule: if frac >= this, auto always picks COLOR (no fallback)
AUTO_FORCE_COLOR_IF_FRAC_GE = 0.02  # 2%

# Output naming
DEFAULT_UNITS_CSV    = "units_POLYGON.csv"
DEFAULT_CENTERS_CSV  = "unit_centers.csv"
DEFAULT_BLOCKS_UNITS = "blocks_units_POLYGON.csv"
DEFAULT_UNITS_OVLY   = "units_overlay.jpg"
DEFAULT_CENTERS_OVLY = "unit_centers_overlay.jpg"
DEFAULT_DBG_MASK     = "debug_color_mask.jpg"
DEFAULT_DBG_THRESH   = "debug_thresh.jpg"
DEFAULT_DECISION_JSON= "extract_decision.json"
DEFAULT_DBG_WHITE_ALL    = "debug_white_all.jpg"
DEFAULT_DBG_WHITE_UNITS  = "debug_white_units.jpg"
DEFAULT_DBG_COMBINED     = "debug_combined_mask.jpg"  # now a side-by-side mosaic

# ===================== Helpers =====================

def scale_by_image(H: int, W: int) -> Tuple[int, int]:
    s = max(H, W) / 4000.0
    return max(1, int(round(ERODE_K0 * s))), int(MIN_UNIT_AREA0 * (s**2))

def poly_area_signed(pts: np.ndarray) -> float:
    x = pts[:, 0].astype(float)
    y = pts[:, 1].astype(float)
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

def ensure_clockwise(pts: np.ndarray) -> np.ndarray:
    return pts[::-1] if poly_area_signed(pts) < 0 else pts

def distance_transform_center(mask_poly: np.ndarray) -> Tuple[int, int]:
    dist = cv2.distanceTransform(mask_poly, cv2.DIST_L2, 3)
    _, _, _, maxLoc = cv2.minMaxLoc(dist)
    return int(maxLoc[0]), int(maxLoc[1])

def mkdir_p(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_csv(df: pd.DataFrame, path: str) -> None:
    if not df.empty:
        df.to_csv(path, index=False)

def points_to_polyline(img: np.ndarray, pts: np.ndarray, color: Tuple[int,int,int], thickness: int = 2, closed_if_ge: int = 3):
    pts_i = pts.reshape(-1, 1, 2).astype(np.int32)
    is_closed = pts.shape[0] >= closed_if_ge
    cv2.polylines(img, [pts_i], isClosed=is_closed, color=color, thickness=thickness)

# ---- CLI HSV parsing helpers ----

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def parse_hsv_triplet(s: str) -> Tuple[int, int, int]:
    h, ss, v = map(int, s.split(","))
    # OpenCV HSV bounds
    h  = _clamp(h, 0, 179)
    ss = _clamp(ss, 0, 255)
    v  = _clamp(v, 0, 255)
    return (h, ss, v)

def parse_hsv_range(spec: str) -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
    # format: H1,S1,V1:H2,S2,V2
    if ":" not in spec:
        raise ValueError(f"Bad --hsv_range '{spec}'. Use H1,S1,V1:H2,S2,V2")
    lo_s, hi_s = spec.split(":", 1)
    lo = parse_hsv_triplet(lo_s)
    hi = parse_hsv_triplet(hi_s)
    return (lo, hi)

def hsv_ranges_from_args(args) -> List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]:
    # Priority: explicit --hsv_range (repeatable) > --color_preset > default HSV_RANGES
    if getattr(args, "hsv_range", None):
        return [parse_hsv_range(r) for r in args.hsv_range]
    if getattr(args, "color_preset", None):
        if args.color_preset not in COLOR_PRESETS:
            raise ValueError(f"Unknown color preset: {args.color_preset}")
        return COLOR_PRESETS[args.color_preset]
    return HSV_RANGES

# ===================== Cross-polygon snapping (optional) =====================

def snap_vertices_crosspoly(df_poly: pd.DataFrame, tol: int = 14) -> pd.DataFrame:
    if df_poly.empty:
        return df_poly
    if not SCIPY_OK:
        print("[WARN] SciPy not available; skipping cross-polygon snapping.")
        return df_poly

    out = df_poly.copy()
    out[["X", "Y"]] = out[["X", "Y"]].astype(int)

    pts = out[["X", "Y"]].values.astype(int)
    meta = list(zip(out["Block"].astype(str), out["Unit"].astype(str)))
    tree = cKDTree(pts)  # type: ignore
    neighbors = tree.query_ball_point(pts, r=tol)

    parent = list(range(len(pts))); rank = [0]*len(pts)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a,b):
        ra,rb = find(a),find(b)
        if ra==rb: return
        if rank[ra]<rank[rb]: parent[ra]=rb
        elif rank[ra]>rank[rb]: parent[rb]=ra
        else: parent[rb]=ra; rank[ra]+=1
    for i,nbrs in enumerate(neighbors):
        for j in nbrs: union(i,j)

    clusters: Dict[int,List[int]] = {}
    for i in range(len(pts)):
        r = find(i); clusters.setdefault(r,[]).append(i)

    rep_map: Dict[int,Tuple[int,int]] = {}
    for r, idxs in clusters.items():
        owners = {(meta[k][0], meta[k][1]) for k in idxs}
        if len(owners) < 2:
            continue
        counts: Dict[Tuple[int,int],int] = {}
        for k in idxs:
            p = (int(pts[k,0]), int(pts[k,1]))
            counts[p] = counts.get(p,0) + 1
        m = max(counts.values())
        cands = [p for p,v in counts.items() if v==m]
        rep = min(cands)
        for k in idxs: rep_map[k]=rep

    if rep_map:
        new = pts.copy()
        for i,rep in rep_map.items(): new[i]=rep
        out["X"], out["Y"] = new[:,0], new[:,1]
        print(f"[CROSSPOLY SNAP] aligned {len(rep_map)} vertices (tol={tol})")
    else:
        print("[CROSSPOLY SNAP] no cross-polygon clusters found")
    return out

# ===================== COLOR MODE =====================

def _extract_units_from_contours(contours, img, eps_frac, min_area, k_erode, color_draw, rows, centers_rows, block_assign_fn):
    """Add unit polygons from given contours into rows & centers_rows and draw overlays."""
    H, W = img.shape[:2]
    overlay_poly = img  # draw in-place on provided image copy
    overlay_ctr  = img
    kept = 0
    unit_id_start = len({r["Unit"] for r in rows if r["Type"] == "Unit"})
    unit_id = unit_id_start
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        eps = max(1.2, eps_frac * cv2.arcLength(c, True))
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) < 3:
            continue
        M = cv2.moments(c)
        cx, cy = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) if M["m00"] else (approx[0,0,0], approx[0,0,1])
        blk = block_assign_fn((cx, cy))
        unit_id += 1
        unit = f"Unit_{unit_id}"
        pts = ensure_clockwise(approx.reshape(-1, 2))
        for i, (x, y) in enumerate(pts, start=1):
            rows.append({"Block": blk, "Type": "Unit", "Unit": unit, "Corner": i, "X": int(x), "Y": int(y)})
        # draw poly
        points_to_polyline(overlay_poly, pts, color_draw)

        # center via distance transform inside polygon
        mask_poly = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask_poly, [pts.reshape(-1,1,2).astype(np.int32)], 255)
        px, py = distance_transform_center(mask_poly)
        centers_rows.append({"Block": blk, "Unit": unit, "CX": px, "CY": py})
        cv2.circle(overlay_ctr, (px, py), 5, (255, 0, 255), -1)
        kept += 1
    return kept

def run_color_mode(image_path: str, out_dir: str) -> Dict[str, Any]:
    img0 = cv2.imread(image_path)
    if img0 is None:
        raise FileNotFoundError(image_path)
    H, W = img0.shape[:2]
    k_erode, MIN_UNIT_AREA = scale_by_image(H, W)

    hsv = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)

    # ---- PRIMARY COLOR MASK (from HSV_RANGES; may be overridden by CLI) ----
    mask_orange = np.zeros((H, W), np.uint8)
    for (lo, hi) in HSV_RANGES:
        lo = np.array(lo, np.uint8); hi = np.array(hi, np.uint8)
        mask_orange |= cv2.inRange(hsv, lo, hi)
    mask_orange = cv2.medianBlur(mask_orange, 3)
    mask_orange = cv2.erode(cv2.medianBlur(mask_orange, 3),
                            cv2.getStructuringElement(cv2.MORPH_RECT, (k_erode, k_erode)), 1)
    cv2.imwrite(os.path.join(out_dir, DEFAULT_DBG_MASK), mask_orange)
    orange_frac = float((mask_orange > 0).mean())

    # ---- WHITE-ISH UNITS (kept separate; background whites removed) ----
    h, s, v = cv2.split(hsv)
    mask_white_all = ((s <= WHITE_S_MAX) & (v >= WHITE_V_MIN)).astype(np.uint8) * 255
    mask_white_all = cv2.morphologyEx(mask_white_all, cv2.MORPH_OPEN,
                                      cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
    # Remove CCs touching image border (page background)
    num_labels, labels = cv2.connectedComponents(mask_white_all, connectivity=4)
    border_labels = set()
    if num_labels > 0:
        border_pixels = np.concatenate([labels[0,:], labels[-1,:], labels[:,0], labels[:,-1]])
        for lab in np.unique(border_pixels):
            if lab != 0:
                border_labels.add(int(lab))
    mask_white_units = np.where(np.isin(labels, list(border_labels)), 0, mask_white_all).astype(np.uint8)
    # Clean + slight shrink
    mask_white_units = cv2.medianBlur(mask_white_units, 3)
    mask_white_units = cv2.erode(mask_white_units,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (k_erode, k_erode)), 1)
    cv2.imwrite(os.path.join(out_dir, DEFAULT_DBG_WHITE_ALL),   mask_white_all)
    cv2.imwrite(os.path.join(out_dir, DEFAULT_DBG_WHITE_UNITS), mask_white_units)

    # ---- DEBUG mosaic: left=primary color, right=white ----
    mosaic = np.zeros((H, W*2), np.uint8)
    mosaic[:, :W]  = mask_orange
    mosaic[:, W:]  = mask_white_units
    cv2.imwrite(os.path.join(out_dir, DEFAULT_DBG_COMBINED), mosaic)

    # Optional: detect big outer blocks (for Block assignment)
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    cnts_blk, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = [c for c in cnts_blk if cv2.contourArea(c) > (H * W * 0.005)]

    def assign_block(pt):
        if not blocks:
            return "Block_1"
        x, y = int(pt[0]), int(pt[1])
        best_i, best_d = 0, 1e12
        for i, c in enumerate(blocks):
            inside = cv2.pointPolygonTest(c, (x, y), False)
            if inside >= 0:
                return f"Block_{i + 1}"
            d = abs(cv2.pointPolygonTest(c, (x, y), True))
            if d < best_d:
                best_d, best_i = d, i
        return f"Block_{best_i + 1}"

    # ---- Contours SEPARATELY on primary color and white ----
    cnts_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_white  , _ = cv2.findContours(mask_white_units, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rows: List[Dict[str, Any]] = []
    centers_rows: List[Dict[str, Any]] = []
    overlay_poly = img0.copy()
    overlay_ctr  = overlay_poly  # same reference for drawing circles

    kept = 0
    kept += _extract_units_from_contours(cnts_orange, overlay_poly, EPS_ARC_FRAC, MIN_UNIT_AREA,
                                         k_erode, (0, 255, 0), rows, centers_rows, assign_block)
    if orange_frac >= WHITE_IN_COLOR_MIN_ORANGE_FRAC:
        kept += _extract_units_from_contours(cnts_white, overlay_poly, EPS_ARC_FRAC, MIN_UNIT_AREA,
                                             k_erode, (0, 255, 255), rows, centers_rows, assign_block)
    # Save outputs
    units_csv   = os.path.join(out_dir, DEFAULT_UNITS_CSV)
    centers_csv = os.path.join(out_dir, DEFAULT_CENTERS_CSV)
    ov_units    = os.path.join(out_dir, DEFAULT_UNITS_OVLY)
    ov_centers  = os.path.join(out_dir, DEFAULT_CENTERS_OVLY)

    write_csv(pd.DataFrame(rows), units_csv)
    write_csv(pd.DataFrame(centers_rows), centers_csv)
    cv2.imwrite(ov_units, overlay_poly)
    cv2.imwrite(ov_centers, overlay_poly)  # centers drawn on same overlay

    return {
        "mode": "color",
        "units_found": int(len({r['Unit'] for r in rows if r['Type']=='Unit'})),
        "orange_pixel_fraction": orange_frac,
        "outputs": {
            "units_csv": units_csv,
            "centers_csv": centers_csv,
            "overlay_units": ov_units,
            "overlay_centers": ov_centers,
            "debug_mask": os.path.join(out_dir, DEFAULT_DBG_MASK),
            "debug_white_all": os.path.join(out_dir, DEFAULT_DBG_WHITE_ALL),
            "debug_white_units": os.path.join(out_dir, DEFAULT_DBG_WHITE_UNITS),
            "debug_combined_mask": os.path.join(out_dir, DEFAULT_DBG_COMBINED),
        },
    }

# ===================== STRUCTURE MODE =====================

def remove_nearby_duplicates(df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    new_rows = []
    for _, group in df.groupby(["Block", "Type", "Unit"]):
        coords: List[Tuple[int, int]] = []
        for _, row in group.iterrows():
            x, y = int(row["X"]), int(row["Y"])
            if all(np.hypot(x - x2, y - y2) > threshold for x2, y2 in coords):
                coords.append((x, y))
                new_rows.append(row)
    return pd.DataFrame(new_rows)

def superclean_points(df: pd.DataFrame, threshold: int = 20) -> pd.DataFrame:
    clean_rows = []
    for _, group in df.groupby(["Block", "Type", "Unit"]):
        coords = group[["X", "Y"]].values
        used = np.zeros(len(coords), dtype=bool)
        for i in range(len(coords)):
            if not used[i]:
                dists = np.hypot(coords[i,0] - coords[:,0], coords[i,1] - coords[:,1])
                used = used | (dists < threshold)
                clean_rows.append(group.iloc[i])
    return pd.DataFrame(clean_rows)

def order_polygon_points(pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] < 3:
        return pts
    if SCIPY_OK and ConvexHull is not None:
        try:
            hull = ConvexHull(pts)  # type: ignore
            return pts[hull.vertices]
        except Exception:
            pass
    visited = np.zeros(pts.shape[0], dtype=bool)
    seq = [0]
    visited[0] = True
    for _ in range(pts.shape[0] - 1):
        last = pts[seq[-1]]
        d = np.linalg.norm(pts - last, axis=1)
        d[visited] = np.inf
        nxt = int(np.argmin(d))
        seq.append(nxt)
        visited[nxt] = True
    return pts[seq]

def centers_for_units(df_poly: pd.DataFrame, H: int, W: int) -> pd.DataFrame:
    centers = []
    mask_poly = np.zeros((H, W), np.uint8)
    for (blk, typ, unit), group in df_poly.groupby(["Block", "Type", "Unit"]):
        if typ != "Unit":
            continue
        pts = group.sort_values("Corner")[['X', 'Y']].values.astype(np.int32)
        if pts.shape[0] < 3:
            continue
        mask_poly[:] = 0
        cv2.fillPoly(mask_poly, [pts.reshape(-1, 1, 2)], 255)
        px, py = distance_transform_center(mask_poly)
        centers.append({"Block": blk, "Unit": unit, "CX": int(px), "CY": int(py)})
    return pd.DataFrame(centers)

def run_structure_mode(image_path: str, out_dir: str, snap_tol: int = 14) -> Dict[str, Any]:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    H, W = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, ADAPT_BLOCK_SIZE, ADAPT_C)
    cv2.imwrite(os.path.join(out_dir, DEFAULT_DBG_THRESH), thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        hierarchy = np.zeros((len(contours), 4), dtype=int)[None, :, :]

    min_block_area = max(1000, int(BLOCK_AREA_FRAC * H * W))
    min_unit_area  = max(200,  int(UNIT_AREA_FRAC  * H * W))

    rows: List[Dict[str, Any]] = []
    block_id = 0

    for cnt, h in zip(contours, hierarchy[0]):
        area = cv2.contourArea(cnt)
        # h: [next, prev, child, parent]
        if h[3] == -1 and area > min_block_area:  # outer shells = blocks
            block_id += 1
            block_label = f"Block_{block_id}"
            epsilon = 0.015 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            for i, pt in enumerate(approx):
                x, y = pt[0][0], pt[0][1]
                rows.append({"Block": block_label, "Type": "Block Boundary", "Unit": "",
                             "Corner": i + 1, "X": int(x), "Y": int(y)})

            # iterate children (units) under this block
            child_idx = h[2]
            unit_id = 0
            while child_idx != -1:
                unit_cnt = contours[child_idx]
                unit_area = cv2.contourArea(unit_cnt)
                if unit_area > min_unit_area:
                    unit_id += 1
                    unit_label = f"Unit_{unit_id}"
                    epsilon_u = 0.015 * cv2.arcLength(unit_cnt, True)
                    approx_u = cv2.approxPolyDP(unit_cnt, epsilon_u, True)
                    for j, pt in enumerate(approx_u):
                        x, y = pt[0][0], pt[0][1]
                        rows.append({"Block": block_label, "Type": "Unit", "Unit": unit_label,
                                     "Corner": j + 1, "X": int(x), "Y": int(y)})
                child_idx = hierarchy[0][child_idx][0]

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Block", "Type", "Unit", "Corner"]).reset_index(drop=True)
        df = remove_nearby_duplicates(df, threshold=10)
        df = superclean_points(df, threshold=20)
        df = df.sort_values(["Block", "Type", "Unit"]).reset_index(drop=True)

    # Ordering + clockwise
    rows_poly: List[Dict[str, Any]] = []
    for (block, typ, unit), group in df.groupby(["Block", "Type", "Unit"]):
        pts = group[["X", "Y"]].values
        ordered = ensure_clockwise(order_polygon_points(pts)) if len(pts) >= 3 else pts
        for i, (x, y) in enumerate(ordered, start=1):
            rows_poly.append({"Block": block, "Type": typ, "Unit": unit,
                              "Corner": i, "X": int(x), "Y": int(y)})

    df_poly = pd.DataFrame(rows_poly)

    # Optional: cross-polygon snapping
    if not df_poly.empty and snap_tol > 0:
        df_poly = snap_vertices_crosspoly(df_poly, tol=int(snap_tol))

    # Save combined & units-only CSVs
    blocks_units_csv = os.path.join(out_dir, DEFAULT_BLOCKS_UNITS)
    write_csv(df_poly, blocks_units_csv)

    units_only = df_poly[df_poly["Type"] == "Unit"].copy()
    units_csv = os.path.join(out_dir, DEFAULT_UNITS_CSV)
    write_csv(units_only, units_csv)

    # Centers for units
    centers_df = centers_for_units(df_poly, H, W)
    centers_csv = os.path.join(out_dir, DEFAULT_CENTERS_CSV)
    write_csv(centers_df, centers_csv)

    # Overlays
    img_units = img.copy()
    img_ctr = img.copy()
    for (block, typ, unit), group in df_poly.groupby(["Block", "Type", "Unit"]):
        pts = group.sort_values("Corner")[['X', 'Y']].values.astype(np.int32)
        color = (255, 0, 0) if typ == "Block Boundary" else (0, 255, 0)
        if pts.shape[0] >= 2:
            points_to_polyline(img_units, pts, color)
    for _, r in centers_df.iterrows():
        cv2.circle(img_ctr, (int(r["CX"]), int(r["CY"])), 5, (255, 0, 255), -1)

    ov_units = os.path.join(out_dir, DEFAULT_UNITS_OVLY)
    ov_centers = os.path.join(out_dir, DEFAULT_CENTERS_OVLY)
    cv2.imwrite(ov_units, img_units)
    cv2.imwrite(ov_centers, img_ctr)

    return {
        "mode": "structure",
        "units_found": int(units_only["Unit"].nunique()) if not units_only.empty else 0,
        "outputs": {
            "blocks_units_csv": blocks_units_csv,
            "units_csv": units_csv,
            "centers_csv": centers_csv,
            "overlay_units": ov_units,
            "overlay_centers": ov_centers,
            "debug_thresh": os.path.join(out_dir, DEFAULT_DBG_THRESH),
        },
    }

# ===================== AUTO MODE DECISION =====================

def quick_color_probe(image_path: str) -> float:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros((H, W), np.uint8)
    for (lo, hi) in HSV_RANGES:
        lo = np.array(lo, np.uint8); hi = np.array(hi, np.uint8)
        mask |= cv2.inRange(hsv, lo, hi)
    return float((mask > 0).mean())

# ===================== Optional exports =====================

def export_debug_points(units_csv_path: str, out_path: str = "vector_debug_points.jpg") -> None:
    df = pd.read_csv(units_csv_path)
    if df.empty:
        print("[export_debug_points] CSV empty; nothing to draw.")
        return
    IMG_W, IMG_H = int(df["X"].max() + 50), int(df["Y"].max() + 50)
    img = np.ones((IMG_H, IMG_W, 3), dtype=np.uint8) * 255
    for x, y in df[["X", "Y"]].values.astype(int):
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    cv2.imwrite(out_path, img)
    print(f"[export_debug_points] saved {out_path}")

def export_points_grouped_json(units_csv_path: str, out_json: str = "points_grouped.json") -> None:
    df = pd.read_csv(units_csv_path)
    if df.empty:
        print("[export_points_grouped_json] CSV empty; nothing to export.")
        return
    groups: List[Dict[str, Any]] = []
    for (block, unit), g in df.groupby(['Block', 'Unit']):
        coords = g.sort_values("Corner")[['X', 'Y']].values.tolist()
        groups.append({"block": str(block), "unit": str(unit), "points": coords})
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)
    print(f"[export_points_grouped_json] saved {out_json}")

# ===================== CLI / Main =====================

def main():
    ap = argparse.ArgumentParser(description="Extract shop polygons via auto-selected pipeline (color or structure).")
    ap.add_argument("--mode", choices=["auto", "color", "structure"], default="auto",
                    help="Force a pipeline or let it auto-select")
    ap.add_argument("--snap_tol", type=int, default=14,
                    help="Snap tolerance (structure mode). Set 0 to disable.")
    ap.add_argument("--export_debug_points", action="store_true",default="vector_debug_points.jpg",
                    help="After extraction, dump all unit points image.")
    ap.add_argument("--export_points_json", action="store_true", default="points_grouped.json",
                    help="After extraction, save grouped points JSON.")

    # NEW: CLI color controls
    ap.add_argument(
        "--color_preset",
        choices=sorted(COLOR_PRESETS.keys()),
        help="Quick color preset for COLOR mode & auto.")
    ap.add_argument(
        "--hsv_range",
        action="append",
        metavar="H1,S1,V1:H2,S2,V2",
        help="Custom HSV inclusive range. Repeatable. Example: --hsv_range 100,50,50:130,255,255")

    args = ap.parse_args()
    load_dotenv(find_dotenv())


    

    # Apply HSV override from args (affects color mode & auto probe)
    try:
        chosen_ranges = hsv_ranges_from_args(args)
    except ValueError as e:
        raise SystemExit(str(e))

    global HSV_RANGES
    HSV_RANGES = chosen_ranges

    # --- ENV-ONLY PATHS (no CLI image) ---
    image_path = os.environ.get("IMAGE_PATH")
    if not image_path:
        raise SystemExit("Set IMAGE_PATH in your .env (no CLI image supported).")
    if not os.path.exists(image_path):
        raise SystemExit(f"IMAGE_PATH does not exist: {image_path}")

    out_dir = os.environ.get("OUT_DIR")
    if not out_dir:
        raise SystemExit("Set OUT_DIR in your .env")
    mkdir_p(out_dir)

    decision: Dict[str, Any] = {"requested_mode": args.mode}

    if args.mode == "color":
        res = run_color_mode(image_path, out_dir)
        decision["selected_mode"] = "color"
        decision.update(res)

    elif args.mode == "structure":
        res = run_structure_mode(image_path, out_dir, snap_tol=args.snap_tol)
        decision["selected_mode"] = "structure"
        decision.update(res)

    else:
        # AUTO
        frac = quick_color_probe(image_path)
        decision["color_pixel_fraction_probe"] = frac

        # STRICT branch: if enough orange, ALWAYS choose COLOR (no fallback)
        if frac >= AUTO_FORCE_COLOR_IF_FRAC_GE:
            res = run_color_mode(image_path, out_dir)
            decision["selected_mode"] = "color"
            decision["auto_reason"] = f"strict_color_frac>={AUTO_FORCE_COLOR_IF_FRAC_GE:.2f}"
            decision.update(res)
        else:
            # Otherwise use standard preference (color if 1%-40%, else structure)
            prefer_color = (COLOR_PIXEL_FRAC_LO <= frac <= COLOR_PIXEL_FRAC_HI)
            first_mode = "color" if prefer_color else "structure"
            res = run_color_mode(image_path, out_dir) if first_mode == "color" \
                else run_structure_mode(image_path, out_dir, snap_tol=args.snap_tol)
            decision["selected_mode"] = first_mode
            decision["auto_reason"] = "standard_preference"
            decision.update(res)

    # Save decision metadata (also record HSV ranges actually used)
    decision["hsv_ranges_used"] = [[list(lo), list(hi)] for (lo, hi) in HSV_RANGES]
    with open(os.path.join(out_dir, DEFAULT_DECISION_JSON), "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    # Optional exports
    units_csv_path = decision.get("outputs", {}).get("units_csv")
    if units_csv_path and os.path.exists(units_csv_path):
        if args.export_debug_points:
            export_debug_points(
                units_csv_path,
                out_path=os.path.join(out_dir, "vector_debug_points.jpg")
            )

        if args.export_points_json:
            export_points_grouped_json(
                units_csv_path,
                out_json=os.path.join(out_dir, "points_grouped.json")
            )


    # Console summary
    print("\n=== Extraction Summary ===")
    core = {k: v for k, v in decision.items()
            if k in ["requested_mode", "selected_mode", "color_pixel_fraction_probe", "units_found", "auto_reason"]}
    print(json.dumps(core, indent=2))
    print("HSV ranges used:", decision.get("hsv_ranges_used"))
    print("Outputs:")
    for k, v in decision.get("outputs", {}).items():
        print(f"  - {k}: {v}")
    

if __name__ == "__main__":
    main()


