# backend/tiny_api.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import json, subprocess, sys, time, logging, os

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tiny_api")

# ---------------- paths ----------------
# layout:  LAYOUT_AI/
#   ├─ backend/tiny_api.py
#   ├─ pipeline/   <- all .py scripts
#   └─ out/        <- all json/png outputs
BASE     = Path(__file__).resolve().parents[1]        # .../LAYOUT_AI
PIPELINE = BASE / "pipeline"
OUT      = BASE / "out"
OUT.mkdir(parents=True, exist_ok=True)

# primary IO files in OUT/
POINTS_JSON    = OUT / "points_with_centroid.json"
EXITS_JSON     = OUT / "centroid_to_corridor_exit.json"
POLYLINES_JSON = OUT / "corridor_polylines.json"
ROUTE_JSON     = OUT / "postman_route.json"

# undo persistence (last batch)
LAST_DELETED_FILE = OUT / ".last_deleted.json"

# Try common filenames for your exit-finder script (prefer normals)
SCRIPT_EXITS = next((p for p in [
    PIPELINE / "centroid_corridor.py",
    PIPELINE / "corridor.py",
] if p.exists()), None)
SCRIPT_BUILD = PIPELINE / "build_corridor.py"
SCRIPT_POSTM = PIPELINE / "postman.py"

# ------------- helpers -------------
def _read_json(p: Path) -> Any:
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _write_json_atomic(p: Path, data: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(p)

def _remove_by_keys_with_removed(
    items: List[Dict[str, Any]],
    keys: set[Tuple[str, str]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (kept_items, removed_items) by strip-matching (block, unit)."""
    def norm(v: Any) -> str:
        return str(v if v is not None else "").strip()
    kept, removed = [], []
    keyset = {(norm(b), norm(u)) for (b, u) in keys}
    for o in items:
        k = (norm(o.get("block")), norm(o.get("unit")))
        (removed if k in keyset else kept).append(o)
    return kept, removed

def _run_script(script: Path, args: List[str], env: Dict[str, str] | None = None) -> None:
    if not script or not script.exists():
        raise RuntimeError(f"Script not found: {script}")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    log.info("Running %s %s", script.name, " ".join(args))
    proc = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(BASE),
        capture_output=True,
        text=True,
        env=merged_env,
    )
    if proc.returncode != 0:
        log.error("%s failed:\nSTDOUT:\n%s\nSTDERR:\n%s", script.name, proc.stdout, proc.stderr)
        raise RuntimeError(
            f"{script.name} failed (exit {proc.returncode})\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )
    if proc.stdout:
        log.debug("%s STDOUT:\n%s", script.name, proc.stdout)
    if proc.stderr:
        log.debug("%s STDERR:\n%s", script.name, proc.stderr)

def _pv() -> int:
    return int(time.time())

def _norm(v: Any) -> str:
    return str(v if v is not None else "").strip()

def _unit_keyset(items: Any) -> set[tuple[str, str]]:
    if not isinstance(items, list):
        return set()
    out: set[tuple[str, str]] = set()
    for u in items:
        if isinstance(u, dict):
            out.add((_norm(u.get("block")), _norm(u.get("unit"))))
    return out

def _file_stat(p: Path) -> dict:
    try:
        st = p.stat()
        return {"exists": True, "size": st.st_size, "mtime": st.st_mtime}
    except FileNotFoundError:
        return {"exists": False, "size": 0, "mtime": None}

# ---------- async rebuild state ----------
_executor = ThreadPoolExecutor(max_workers=1)
_state_lock = Lock()
_undo_lock = Lock()
_undo_stack: List[List[Dict[str, Any]]] = []   # in-memory history (batches)

_state: Dict[str, Any] = {
    "running": False,
    "stage": None,
    "lastLog": "",
    "lastError": None,
    "startedAt": None,
    "finishedAt": None,
    "projectVersion": None,
}

def _set_state(**kw):
    with _state_lock:
        _state.update(kw)

def _save_last_deleted(batch: List[Dict[str, Any]]) -> None:
    try:
        _write_json_atomic(LAST_DELETED_FILE, {"batch": batch})
    except Exception as e:
        log.warning("Failed to save last-deleted file: %s", e)

def _load_last_deleted() -> List[Dict[str, Any]]:
    try:
        js = _read_json(LAST_DELETED_FILE) or {}
        return list(js.get("batch") or [])
    except Exception:
        return []

def _guard_points_or_restore(snapshot: list[dict], tag: str) -> None:
    """Ensure points JSON matches snapshot (deletions kept). If changed, restore."""
    now = _read_json(POINTS_JSON) or []
    if _unit_keyset(now) != _unit_keyset(snapshot):
        log.warning("Points changed after step %s (before=%d, after=%d) — restoring snapshot.",
                    tag, len(snapshot), len(now))
        _write_json_atomic(POINTS_JSON, snapshot)

def _rebuild_all_job():
    try:
        _set_state(running=True, stage="start", lastError=None,
                   startedAt=time.time(), finishedAt=None, lastLog="Starting rebuild…")
        log.info("Rebuild started")

        # Snapshot of points
        points_snap = _read_json(POINTS_JSON) or []
        log.info("Snapshot units: %d", len(points_snap))

        # Common env for all children (ensure OUT_DIR propagates)
        common_env = {
            "OUT_DIR": str(OUT),
            "LAYOUT_POINTS_JSON": str(POINTS_JSON),
            "LAYOUT_EXITS_JSON": str(EXITS_JSON),
            "LAYOUT_POLYLINES_JSON": str(POLYLINES_JSON),
            "LAYOUT_ROUTE_JSON": str(ROUTE_JSON),
            "LAYOUT_OUTDIR": str(OUT),
        }

        # 1) Exits (if script present) — NO CLI; scripts read OUT_DIR/JSONs themselves
        if SCRIPT_EXITS:
            _set_state(stage="exits", lastLog="Recomputing exits…")
            try:
                _run_script(SCRIPT_EXITS, [], env=common_env)
            finally:
                _guard_points_or_restore(points_snap, "exits")

        # 2) Corridor polylines (build_corridor.py) — explicit IO paths
        _set_state(stage="polylines", lastLog="Building corridor polylines…")
        try:
            _run_script(
                SCRIPT_BUILD,
                [
                    "--units", str(POINTS_JSON),
                    "--exits", str(EXITS_JSON),
                    "--outdir", str(OUT),
                    "--downscale", "3",
                    "--clearance", "8",
                    "--stitch-dist", "18",
                    "--stitch-angle", "30",
                    "--simplify-eps", "1.2",
                ],
                env=common_env,
            )
        finally:
            _guard_points_or_restore(points_snap, "polylines")

        # 3) Single route (postman.py)
        _set_state(stage="route", lastLog="Computing single route…")
        try:
            _run_script(
                SCRIPT_POSTM,
                [
                    "--polylines", str( POLYLINES_JSON ),
                    "--outdir", str(OUT),
                    "--snap", "12",
                    "--compress-iters", "6",
                    "--connect", "shortest",
                    "--style", "straight",
                    "--prune-spurs", "0",
                    "--simplify-eps", "0",
                ],
                env=common_env,
            )
        finally:
            _guard_points_or_restore(points_snap, "route")

        # Keep snapshot as ground-truth
        _write_json_atomic(POINTS_JSON, points_snap)

        _set_state(stage="done", lastLog="Rebuild done", projectVersion=_pv())
        log.info("Rebuild finished")

    except Exception as e:
        _set_state(lastError=str(e), lastLog=f"Error: {e}", stage="error")
        log.exception("Rebuild error")
    finally:
        _set_state(running=False, finishedAt=time.time())

def _start_rebuild_if_needed() -> bool:
    with _state_lock:
        if _state.get("running"):
            log.info("Rebuild already running; skip")
            return False
        _state.update(running=True, stage="queued", lastError=None,
                      startedAt=time.time(), finishedAt=None, lastLog="Queued for rebuild…")
    _executor.submit(_rebuild_all_job)
    return True

# ------------- models -------------
class UnitKey(BaseModel):
    block: str = Field(...)
    unit:  str = Field(...)

class DeletePayload(BaseModel):
    items: List[UnitKey]
    rebuild: bool = True   # ignored (delete never starts rebuild)

class RestorePayload(BaseModel):
    items: List[Dict[str, Any]]  # full unit objects (as in points_with_centroid.json)

# ------------- app -------------
app = FastAPI(title="Tiny Local Layout Backend", version="1.8")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "ts": _pv()}

@app.get("/version")
def version():
    return {"projectVersion": _pv()}

@app.get("/rebuild/status")
def rebuild_status():
    with _state_lock:
        units = _read_json(POINTS_JSON)
        exits = _read_json(EXITS_JSON)
        polyl = _read_json(POLYLINES_JSON)
        route = _read_json(ROUTE_JSON)
        return {
            "running": _state["running"],
            "stage": _state["stage"],
            "lastLog": _state["lastLog"],
            "lastError": _state["lastError"],
            "startedAt": _state["startedAt"],
            "finishedAt": _state["finishedAt"],
            "projectVersion": _state["projectVersion"],
            "files": {
                "points": _file_stat(POINTS_JSON)    | {"count": len(units or [])},
                "exits":  _file_stat(EXITS_JSON)     | {"count": len(exits or [])},
                "polyl":  _file_stat(POLYLINES_JSON) | {"count": len((polyl or {}).get("polylines", []))},
                "route":  _file_stat(ROUTE_JSON)     | {"count": len((route or {}).get("line", []))},
            }
        }

@app.post("/rebuild")
def manual_rebuild():
    started = _start_rebuild_if_needed()
    return {"ok": True, "started": started, "projectVersion": _pv()}

@app.delete("/units")
def delete_units(payload: DeletePayload):
    if not payload.items:
        raise HTTPException(400, "No items provided")

    keys = {(i.block.strip(), i.unit.strip()) for i in payload.items}

    units = _read_json(POINTS_JSON)
    if units is None or not isinstance(units, list):
        raise HTTPException(500, f"{POINTS_JSON.name} is missing or not a list")

    new_units, removed_items = _remove_by_keys_with_removed(units, keys)
    if removed_items:
        _write_json_atomic(POINTS_JSON, new_units)
        with _undo_lock:
            _undo_stack.append(removed_items)
            if len(_undo_stack) > 20:
                _undo_stack.pop(0)
        _save_last_deleted(removed_items)
        log.info("Deleted %d unit(s) from points JSON", len(removed_items))
    else:
        log.info("No units matched for deletion")

    return {
        "removed": {"units": len(removed_items)},
        "rebuildStarted": False,   # delete never triggers rebuild
        "projectVersion": _pv(),
    }

# Optional POST fallback for clients that can't send body with DELETE
@app.post("/units/delete")
def delete_units_post(payload: DeletePayload):
    return delete_units(payload)

@app.post("/units/restore")
def restore_units(payload: RestorePayload):
    units = _read_json(POINTS_JSON)
    if units is None or not isinstance(units, list):
        raise HTTPException(500, f"{POINTS_JSON.name} is missing or not a list")

    existing = {(_norm(u.get("block")), _norm(u.get("unit"))) for u in units}
    added = 0
    for u in payload.items:
        b, n = _norm(u.get("block")), _norm(u.get("unit"))
        if not b or not n:
            continue
        if (b, n) in existing:
            continue
        units.append(u)
        existing.add((b, n))
        added += 1

    if added:
        _write_json_atomic(POINTS_JSON, units)
        log.info("Restored %d unit(s) into points JSON", added)

    return {"added": added, "projectVersion": _pv()}

@app.post("/units/undo")
def undo_last_delete():
    """Restore last deleted batch (in-memory stack first, else disk fallback)."""
    batch: List[Dict[str, Any]] = []
    with _undo_lock:
        if _undo_stack:
            batch = _undo_stack.pop()
        else:
            batch = _load_last_deleted()

    if not batch:
        return {"restored": 0, "projectVersion": _pv()}

    units = _read_json(POINTS_JSON) or []
    existing = {(_norm(u.get("block")), _norm(u.get("unit"))) for u in units}
    restored = 0
    for u in batch:
        b, n = _norm(u.get("block")), _norm(u.get("unit"))
        if not b or not n: 
            continue
        if (b, n) in existing:
            continue
        units.append(u)
        existing.add((b, n))
        restored += 1

    if restored:
        _write_json_atomic(POINTS_JSON, units)
        log.info("Undo restored %d unit(s)", restored)

    # clear disk copy only if we restored from disk
    if not _undo_stack and LAST_DELETED_FILE.exists():
        try:
            LAST_DELETED_FILE.unlink()
        except Exception:
            pass

    return {"restored": restored, "projectVersion": _pv()}

# ---- static mounts ----
# Serve generated jsons from /out, and app files (index.html etc.) from /pipeline
app.mount("/out", StaticFiles(directory=str(OUT)), name="out")
app.mount("/", StaticFiles(directory=str(PIPELINE), html=True), name="public")
IMAGES = BASE / "images"     # ya PIPELINE / "images" if that’s where the files live
app.mount("/images", StaticFiles(directory=str(IMAGES)), name="images")
app.mount("/out",    StaticFiles(directory=str(OUT)),    name="out")
app.mount("/",       StaticFiles(directory=str(PIPELINE), html=True), name="public")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tiny_api:app", host="127.0.0.1", port=8000, reload=True)
