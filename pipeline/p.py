# p.py — simple orchestrator (no argparse, runs main.py with args)
from __future__ import annotations
import os, subprocess, sys, time
from pathlib import Path
from dotenv import load_dotenv, find_dotenv  # pip install python-dotenv

# ---- editable defaults ----
# main.py args (tweak these)
MAIN_MODE         = "auto"      # "auto" | "color" | "structure"
MAIN_SNAP_TOL     = 14          # int; 0 disables snapping (structure mode)
MAIN_COLOR_PRESET = None        # e.g. "orange", "blue", "red" ... or None
MAIN_HSV_RANGES   = []          # e.g. ["100,50,50:130,255,255"] (repeatable)

# pipeline toggles
SKIP_MAIN_EXTRACT = False   # True -> skip main.py entirely
SKIP_OCR       = False      # True -> skip OCR + matching
SKIP_CENTROID  = False      # True -> skip centroid step
SKIP_EXITS     = False      # True -> skip corridor exits step

# post-processing
CONNECT_MODE   = "shortest"  # "shortest" | "none"
POSTMAN_STYLE  = "straight"  # "straight" | "geom"
SNAP_TOL       = 14
COMPRESS_ITERS = 6
# ---------------------------

ROOT = Path(__file__).resolve().parent      # ...\LAYOUT_AI\pipeline
PY   = sys.executable

def run_step(title: str, cmd: list[str], cwd: Path) -> None:
    print(f"\n=== {title} ===")
    print(">>", " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(cwd))
    dt = time.time() - t0
    if r.returncode != 0:
        raise SystemExit(f"[FAIL] {title} (exit {r.returncode})")
    print(f"[OK] {title} ({dt:.1f}s)")

def main():
    # Load .env and resolve OUT_DIR (fallback to project-root /out)
    load_dotenv(find_dotenv())
    out_dir = Path(os.getenv("OUT_DIR") or (ROOT.parent / "out"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make sure children also see OUT_DIR
    os.environ["OUT_DIR"] = str(out_dir)

    # FIX: p.py is already inside pipeline, so use ROOT
    pipeline_dir = ROOT

    # OUT_DIR inputs/outputs for downstream steps
    points_json = out_dir / "points_with_centroid.json"
    exits_json  = out_dir / "centroid_to_corridor_exit.json"
    poly_json   = out_dir / "corridor_polylines.json"

    # 0) main.py (actual run with args)
    main_py = pipeline_dir / "main.py"
    if not SKIP_MAIN_EXTRACT and main_py.exists():
        cmd = [PY, str(main_py)]
        if MAIN_MODE:
            cmd += ["--mode", str(MAIN_MODE)]
        if isinstance(MAIN_SNAP_TOL, (int, float)):
            cmd += ["--snap_tol", str(int(MAIN_SNAP_TOL))]
        if MAIN_COLOR_PRESET:
            cmd += ["--color_preset", str(MAIN_COLOR_PRESET)]
        for rng in MAIN_HSV_RANGES or []:
            cmd += ["--hsv_range", str(rng)]
        run_step("Layout extraction (main.py)", cmd, cwd=pipeline_dir)
    elif not main_py.exists():
        print("[warn] main.py not found; skipping main extraction")
    else:
        print("[skip] main extraction")

    # 1) OCR + matching
    if not SKIP_OCR:
        run_step("OCR extract",      [PY, str(pipeline_dir / "ocr_extract.py")], cwd=pipeline_dir)
        run_step("Match OCR->polys", [PY, str(pipeline_dir / "match_ocr_to_polygons.py")], cwd=pipeline_dir)
    else:
        print("[skip] OCR + matching")

    # 2) Centroids
    if not SKIP_CENTROID:
        run_step("Centroids", [PY, str(pipeline_dir / "centroid.py")], cwd=pipeline_dir)
    else:
        print("[skip] centroids")

    # 3) Corridor exits
    if not SKIP_EXITS:
        exits_script = None
        for name in ("corridor.py", "centroid_corridor.py", "corridor_normals.py"):
            p = pipeline_dir / name
            if p.exists():
                exits_script = p
                break
        if exits_script:
            run_step("Corridor exits", [PY, str(exits_script)], cwd=pipeline_dir)
        else:
            print("[warn] No exits script found (corridor.py / centroid_corridor.py / corridor_normals.py). Continuing…")
    else:
        print("[skip] corridor exits")

    # 4) Build corridor polylines (read/write in OUT_DIR)
    run_step("Build corridor polylines", [
        PY, str(pipeline_dir / "build_corridor.py"),
        "--units", str(points_json),
        "--exits", str(exits_json),
        "--outdir", str(out_dir)
    ], cwd=pipeline_dir)

    # 5) Postman route (read/write in OUT_DIR)
    run_step("Postman route", [
        PY, str(pipeline_dir / "postman.py"),
        "--polylines", str(poly_json),
        "--outdir", str(out_dir),
        "--snap", str(SNAP_TOL),
        "--compress-iters", str(COMPRESS_ITERS),
        "--connect", CONNECT_MODE,
        "--style", POSTMAN_STYLE,
    ], cwd=pipeline_dir)

    print("\nALL DONE ✅")
    print("OUT_DIR:", out_dir)

if __name__ == "__main__":
    main()
