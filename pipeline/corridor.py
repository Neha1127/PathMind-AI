# corridor_normals.py  — env-only I/O (OUT_DIR)
import os, sys, json, math
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from shapely.geometry import Polygon, Point, LineString
from shapely.prepared import prep

# Load .env and resolve OUT_DIR
load_dotenv(find_dotenv())
OUT_DIR = Path(os.getenv("OUT_DIR") or Path(__file__).resolve().parent / "out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Default paths inside OUT_DIR
DEFAULT_INPUT  = OUT_DIR / "points_with_centroid.json"
DEFAULT_OUTPUT = OUT_DIR / "centroid_to_corridor_exit.json"


def _outward_edge_normals(poly: Polygon):
    """
    Return unique outward unit normals (nx, ny) for every edge of poly.
    Outwardness is decided by sampling a tiny step from each edge's midpoint.
    """
    coords = list(poly.exterior.coords[:-1])  # drop repeated last point
    normals = []
    seen = set()

    for i in range(len(coords)):
        (x1, y1) = coords[i]
        (x2, y2) = coords[(i + 1) % len(coords)]
        ex, ey = (x2 - x1), (y2 - y1)
        # left normal to the edge direction
        nx, ny = -ey, ex
        length = math.hypot(nx, ny)
        if length == 0:
            continue
        nx, ny = nx / length, ny / length

        # Decide inward/outward by sampling from the edge midpoint
        midx, midy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        eps = 1e-6
        sample_out = Point(midx + nx * eps, midy + ny * eps)

        # If sample is inside/touching, normal is pointing inward → flip
        if poly.contains(sample_out) or poly.touches(sample_out):
            nx, ny = -nx, -ny

        # Dedup nearly-parallel normals (avoid spam for long runs of parallel edges)
        ang = round(math.atan2(ny, nx), 3)  # ~0.17° buckets
        if ang in seen:
            continue
        seen.add(ang)
        normals.append((nx, ny))

    return normals


def find_corridor_exit_perp_normals(
    json_path: str | Path | None = None,
    output_path: str | Path | None = None,
    step_out: int = 20,
    max_steps: int = 120,
    steps_after_boundary: int = 3,
):
    # Resolve paths in OUT_DIR by default
    json_path = Path(json_path) if json_path else DEFAULT_INPUT
    output_path = Path(output_path) if output_path else DEFAULT_OUTPUT

    if not json_path.exists():
        sys.exit(f"Missing input: {json_path} (expected in OUT_DIR)")

    with open(json_path, "r", encoding="utf-8") as f:
        units = json.load(f)

    polygons = [Polygon(unit["polygon_points"]) for unit in units]
    prepped = [prep(p) for p in polygons]  # speed up contains/intersects
    result = []

    for idx, unit in enumerate(units):
        poly = Polygon(unit["polygon_points"])
        centroid = Point(unit["centroid"])

        best = None
        min_dist = float("inf")

        # 1) build outward, edge-perpendicular directions
        directions = _outward_edge_normals(poly)

        # 2) try each outward normal
        for (dx, dy) in directions:
            # Step out to boundary in this direction
            dist = 0
            just_exited = None
            for k in range(1, max_steps):
                dist = k * step_out
                test_pt = Point(centroid.x + dx * dist, centroid.y + dy * dist)
                if not poly.contains(test_pt):
                    just_exited = test_pt
                    break
            if just_exited is None:
                continue

            # Step a few more to ensure we're outside ALL units
            corridor_candidate = None
            for extra in range(1, steps_after_boundary + 1):
                extra_dist = dist + extra * step_out
                out_pt = Point(centroid.x + dx * extra_dist, centroid.y + dy * extra_dist)
                if not any(prepped[pidx].contains(out_pt) for pidx in range(len(polygons))):
                    corridor_candidate = out_pt
                    break

            if not corridor_candidate:
                continue

            # Line-of-sight clear? (avoid crossing/touching any other polygon)
            line = LineString([centroid, corridor_candidate])
            valid = True
            for pidx, p in enumerate(polygons):
                if pidx == idx:
                    continue
                if line.crosses(p) or line.touches(p) or line.within(p):
                    valid = False
                    break

            if not valid:
                continue

            cand_dist = centroid.distance(corridor_candidate)
            if cand_dist < min_dist:
                min_dist = cand_dist
                best = corridor_candidate

        result.append({
            "block": unit.get("block"),
            "unit": unit.get("unit"),
            "centroid": unit.get("centroid"),
            "exit_point": [best.x, best.y] if best else None,
            "is_corridor": bool(best),
            "best_text": unit.get("best_text", ""),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    # No args → uses OUT_DIR defaults
    find_corridor_exit_perp_normals(
        step_out=20,
        max_steps=120,
        steps_after_boundary=3,
    )
