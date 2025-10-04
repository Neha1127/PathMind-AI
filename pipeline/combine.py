#!/usr/bin/env python3
"""
combine.py — Merge 4 corridor JSONs into one out/indata.json.

Defaults (relative to current working directory):
  --units       points_with_centroid.json
  --exits       centroid_to_corridor_exit.json
  --polylines   out/corridor_polylines.json
  --route       out/postman_route.json
  --out         out/indata.json

Run:
  python combine.py            # fast, compact JSON
  python combine.py --pretty   # human-friendly formatting
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, data: Any, *, indent: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge 4 JSONs into a single indata.json")
    ap.add_argument("--units", type=Path, default=Path("points_with_centroid.json"))
    ap.add_argument("--exits", type=Path, default=Path("centroid_to_corridor_exit.json"))
    ap.add_argument("--polylines", type=Path, default=Path("out/corridor_polylines.json"))
    ap.add_argument("--route", type=Path, default=Path("out/postman_route.json"))
    ap.add_argument("--out", type=Path, default=Path("out/indata.json"))
    ap.add_argument("--project-version", type=int, default=None, help="Override projectVersion (int). Default = now().")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON with indent=2")
    args = ap.parse_args()

    # Read inputs
    units = read_json(args.units)
    if not isinstance(units, list):
        raise TypeError(f"{args.units.name} must be a JSON array (list); got {type(units).__name__}")

    exits = read_json(args.exits)
    polys = read_json(args.polylines)
    route = read_json(args.route)

    # Accept either {"polylines":[...]} or a raw array [...]
    polylines = polys.get("polylines") if isinstance(polys, dict) and "polylines" in polys else polys

    bundle = {
        "projectVersion": int(args.project_version if args.project_version is not None else time.time()),
        "units": units,
        "exits": exits,
        "polylines": polylines,
        "route": route,
    }

    write_json_atomic(args.out, bundle, indent=2 if args.pretty else None)

    # Friendly summary
    units_count = len(units)
    exits_count = len(exits) if isinstance(exits, list) else (len(exits) if hasattr(exits, "__len__") else "n/a")
    poly_count = len(polylines) if isinstance(polylines, list) else "n/a"
    route_points = len(route.get("line", [])) if isinstance(route, dict) else "n/a"

    print(f"Bundled → {args.out}")
    print(f"  units={units_count}, exits={exits_count}, polylines={poly_count}, route_points={route_points}")
    print(f"  projectVersion={bundle['projectVersion']}")


if __name__ == "__main__":
    main()
