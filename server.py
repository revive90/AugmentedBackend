import sys, os, json, subprocess
from pathlib import Path
from typing import List, Optional, Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:5173", "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SERVE_ROOT = Path(__file__).parent.resolve()
app.mount("/static", StaticFiles(directory=str(SERVE_ROOT)), name="static")

# ----------------- Request models -----------------
class BaselineRequest(BaseModel):
    ROOT_DATASET_DIR: str
    MAIN_OUTPUT_DIR: str
    INITIAL_TH1: float
    INITIAL_TH2: float
    ACCEPTABLE_DIFFERENCE_PERCENTAGE: float

# Threshold mode (used by /augment/enhanced/stream in  threshold page)
class EnhancedThresholdRequest(BaseModel):
    ROOT_DATASET_DIR: str
    AUGMENTED_OUTPUT_DIR: str
    LOWER_THRESHOLD: float
    UPPER_THRESHOLD: float

# Percentage mode (used by EnhancedPerc.tsx)
class EnhancedPercentageRequest(BaseModel):
    ROOT_DATASET_DIR: str
    AUGMENTED_OUTPUT_DIR: str
    AUGMENTATION_TARGET_PERCENTAGE: float

# Class-specific mode (used by EnhancedClassSpecific.tsx)
class EnhancedClassSpecificRequest(BaseModel):
    ROOT_DATASET_DIR: str
    AUGMENTED_OUTPUT_DIR: str
    CLASS_TARGETS_JSON: str  # JSON string e.g. {"glioma":2000,...}

# ----------------- Helpers -----------------
def to_static_url(p: Path) -> Optional[str]:
    try:
        rel = p.resolve().relative_to(SERVE_ROOT)
        return f"/static/{rel.as_posix()}"
    except Exception:
        return None

def _latest_summary_url(script_dir: Path) -> Optional[str]:
    """Fallback: try to surface the most recent summary file."""
    results = script_dir / "augmentation_results"
    if not results.exists():
        return None
    patterns = [
        "*_augmentation_summary.txt",
        "augmentation_newPipeline_results_*.txt",
        "augmentation_threshold_mode_*.txt",
        "augmentation_percentage_mode_*.txt",
        "augmentation_class_specific_mode_*.txt",
    ]
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(results.glob(pat))
    if not candidates:
        return None
    latest = sorted(candidates, key=os.path.getmtime)[-1]
    url = to_static_url(latest)
    if url:
        return url
    dest = SERVE_ROOT / "static_previews" / latest.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not dest.exists():
            dest.write_bytes(latest.read_bytes())
        return to_static_url(dest)
    except Exception:
        return None

def _path_to_static_url(p: Path) -> Optional[str]:
    try:
        url = to_static_url(p)
        if url:
            return url
    except Exception:
        pass
    try:
        dest = SERVE_ROOT / "static_previews" / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(p.read_bytes())
        return to_static_url(dest)
    except Exception:
        return None

def pick_sample_images_from_classes(out_dir: Path, limit: int = 5) -> List[str]:
    if not out_dir.exists():
        return []
    class_dirs = [d for d in out_dir.iterdir() if d.is_dir()]
    per_class_files: List[List[Path]] = []
    exts = {".png", ".jpg", ".jpeg"}
    for c in class_dirs:
        files = sorted([p for p in c.rglob("*") if p.suffix.lower() in exts])
        if files:
            per_class_files.append(files)
    if not per_class_files:
        return []

    picks: List[Path] = []
    idx = 0
    while len(picks) < limit and per_class_files:
        for files in list(per_class_files):
            if idx < len(files):
                picks.append(files[idx])
                if len(picks) >= limit:
                    break
            else:
                per_class_files.remove(files)
        idx += 1

    urls: List[str] = []
    previews_root = SERVE_ROOT / "static_previews"
    previews_root.mkdir(parents=True, exist_ok=True)
    for p in picks:
        url = to_static_url(p)
        if url:
            urls.append(url)
            continue
        dest = previews_root / p.name
        try:
            if not dest.exists():
                dest.write_bytes(p.read_bytes())
            rel = dest.resolve().relative_to(SERVE_ROOT)
            urls.append(f"/static/{rel.as_posix()}")
        except Exception:
            pass
    return urls

# ---------- Safety check for folder deletion----------
def _validate_path(path_str: str, label: str) -> Path:
    if not path_str.strip():
        raise HTTPException(400, f"{label} must not be empty.")
    path = Path(path_str).resolve()
    if path == Path("/"):
        raise HTTPException(400, f"{label} path is invalid or too broad.")
    return path

# ---------- Process streaming with EVT latching ----------
SENTINEL = "[[__DONE__]]"

def _stream_proc(cmd: List[str], cwd: Path, out_dir: Path, script_dir: Path) -> Iterator[str]:
    """
    Stream stdout and *capture* [[EVT]] 'done' payload fields (summary paths, total_generated)
    so its can return the precise summary for THIS run (no cross-run mixups).
    """
    last_summary_txt: Optional[str] = None
    last_summary_json: Optional[str] = None
    last_total_generated: Optional[int] = None

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=str(cwd), encoding="utf-8", errors="replace"
    )
    try:
        if proc.stdout is not None:
            for raw in proc.stdout:
                line = raw.replace("\r", "\n")
                # Echo through
                yield line
                # Capture EVT
                if line.startswith("[[EVT]]"):
                    try:
                        evt_json = line[len("[[EVT]]"):].strip()
                        evt = json.loads(evt_json)
                        if evt.get("type") == "done":
                            if "summary_path" in evt:
                                last_summary_txt = evt["summary_path"]
                            if "summary_json" in evt:
                                last_summary_json = evt["summary_json"]
                            if "total_generated" in evt:
                                try:
                                    last_total_generated = int(evt["total_generated"])
                                except Exception:
                                    pass
                    except Exception:
                        pass
    finally:
        proc.wait()

    samples = pick_sample_images_from_classes(out_dir, limit=5)

    # Prefer the summary from THIS run
    summary_url = None
    if last_summary_txt:
        p = Path(last_summary_txt)
        if p.exists():
            summary_url = _path_to_static_url(p)

    # Fallback: latest summary in folder
    if not summary_url:
        summary_url = _latest_summary_url(script_dir)

    payload = {"done": True, "samples": samples, "summary_url": summary_url}
    if last_total_generated is not None:
        payload["total_generated"] = last_total_generated

    yield f"\n{SENTINEL} " + json.dumps(payload) + "\n"

# ----------------- Utility endpoints -----------------
@app.get("/datasets/scan-classes")
def scan_classes(root: str = Query(..., description="Path to dataset root")):
    root_dir = _validate_path(root, "ROOT_DATASET_DIR")
    if not root_dir.exists():
        raise HTTPException(400, f"ROOT_DATASET_DIR not found: {root_dir}")
    classes = []
    for d in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        count = 0
        for ext in (".png", ".jpg", ".jpeg"):
            count += len(list(d.rglob(f"*{ext}")))
        classes.append({"name": d.name, "image_count": count})
    return {"root": str(root_dir), "classes": classes}

# ----------------- Baseline endpoints -----------------
def _build_cmd_baseline(script: Path, root_dir: Path, out_dir: Path, th1: float, th2: float, tol: float) -> List[str]:
    return [
        sys.executable, "-u", str(script),
        "--root-dataset-dir", str(root_dir),
        "--main-output-dir", str(out_dir),
        "--initial-th1", str(th1),
        "--initial-th2", str(th2),
        "--acceptable-difference-percentage", str(tol),
    ]

@app.post("/augment/baseline")
def run_baseline(req: BaselineRequest):
    script = (Path(__file__).parent / "ocmri_baseline.py").resolve()
    root_dir = _validate_path(req.ROOT_DATASET_DIR, "ROOT_DATASET_DIR")
    out_dir = _validate_path(req.MAIN_OUTPUT_DIR, "MAIN_OUTPUT_DIR")

    if not script.exists():
        raise HTTPException(500, f"Script not found: {script}")
    if not root_dir.exists():
        raise HTTPException(400, f"ROOT_DATASET_DIR not found: {root_dir}")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd_baseline(script, root_dir, out_dir, req.INITIAL_TH1, req.INITIAL_TH2, req.ACCEPTABLE_DIFFERENCE_PERCENTAGE)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(script.parent))
    samples = pick_sample_images_from_classes(out_dir, limit=5)
    summary_url = _latest_summary_url(script.parent)
    return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "samples": samples, "summary_url": summary_url}

@app.post("/augment/baseline/stream")
def run_baseline_stream(req: BaselineRequest):
    script = (Path(__file__).parent / "ocmri_baseline.py").resolve()
    root_dir = _validate_path(req.ROOT_DATASET_DIR, "ROOT_DATASET_DIR")
    out_dir = _validate_path(req.MAIN_OUTPUT_DIR, "MAIN_OUTPUT_DIR")

    if not script.exists():
        raise HTTPException(500, f"Script not found: {script}")
    if not root_dir.exists():
        raise HTTPException(400, f"ROOT_DATASET_DIR not found: {root_dir}")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd_baseline(script, root_dir, out_dir, req.INITIAL_TH1, req.INITIAL_TH2, req.ACCEPTABLE_DIFFERENCE_PERCENTAGE)
    gen = _stream_proc(cmd, cwd=script.parent, out_dir=out_dir, script_dir=script.parent)
    resp = StreamingResponse(gen, media_type="text/plain")
    resp.headers["X-Accel-Buffering"] = "no"
    return resp

# ----------------- Threshold mode -----------------
def _build_cmd_threshold(script: Path, root_dir: Path, out_dir: Path, lower: float, upper: float) -> List[str]:
    return [
        sys.executable, "-u", str(script),
        "--root-dataset-dir", str(root_dir),
        "--augmented-output-dir", str(out_dir),
        "--lower-threshold", str(lower),
        "--upper-threshold", str(upper),
    ]

@app.post("/augment/enhanced/stream")  #  threshold page hits this ep
def run_enhanced_threshold_stream(req: EnhancedThresholdRequest):
    script = (Path(__file__).parent / "run_threshold_pipeline.py").resolve()
    root_dir = _validate_path(req.ROOT_DATASET_DIR, "ROOT_DATASET_DIR")
    out_dir = _validate_path(req.AUGMENTED_OUTPUT_DIR, "AUGMENTED_OUTPUT_DIR")

    if not script.exists():
        raise HTTPException(500, f"Script not found: {script}")
    if not root_dir.exists():
        raise HTTPException(400, f"ROOT_DATASET_DIR not found: {root_dir}")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd_threshold(script, root_dir, out_dir, req.LOWER_THRESHOLD, req.UPPER_THRESHOLD)
    gen = _stream_proc(cmd, cwd=script.parent, out_dir=out_dir, script_dir=script.parent)
    resp = StreamingResponse(gen, media_type="text/plain")
    resp.headers["X-Accel-Buffering"] = "no"
    return resp

# ----------------- Percentage mode -----------------
def _build_cmd_percentage(script: Path, root_dir: Path, out_dir: Path, pct: float) -> List[str]:
    return [
        sys.executable, "-u", str(script),
        "--root-dataset-dir", str(root_dir),
        "--augmented-output-dir", str(out_dir),
        "--augmentation-target-percentage", str(pct),
    ]

@app.post("/augment/enhanced/percentage/stream")
def run_enhanced_percentage_stream(req: EnhancedPercentageRequest):
    script = (Path(__file__).parent / "run_percentage_pipeline.py").resolve()
    root_dir = _validate_path(req.ROOT_DATASET_DIR, "ROOT_DATASET_DIR")
    out_dir = _validate_path(req.AUGMENTED_OUTPUT_DIR, "AUGMENTED_OUTPUT_DIR")

    if not script.exists():
        raise HTTPException(500, f"Script not found: {script}")
    if not root_dir.exists():
        raise HTTPException(400, f"ROOT_DATASET_DIR not found: {root_dir}")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd_percentage(script, root_dir, out_dir, req.AUGMENTATION_TARGET_PERCENTAGE)
    gen = _stream_proc(cmd, cwd=script.parent, out_dir=out_dir, script_dir=script.parent)
    resp = StreamingResponse(gen, media_type="text/plain")
    resp.headers["X-Accel-Buffering"] = "no"
    return resp

# ----------------- Class-specific mode -----------------
def _build_cmd_class(script: Path, root_dir: Path, out_dir: Path, targets_json: str) -> List[str]:
    return [
        sys.executable, "-u", str(script),
        "--root-dataset-dir", str(root_dir),
        "--augmented-output-dir", str(out_dir),
        "--class-targets-json", targets_json,
    ]

@app.post("/augment/enhanced/class-specific/stream")
def run_enhanced_class_specific_stream(req: EnhancedClassSpecificRequest):
    script = (Path(__file__).parent / "run_class_specific_pipeline.py").resolve()
    root_dir = _validate_path(req.ROOT_DATASET_DIR, "ROOT_DATASET_DIR")
    out_dir = _validate_path(req.AUGMENTED_OUTPUT_DIR, "AUGMENTED_OUTPUT_DIR")

    if not script.exists():
        raise HTTPException(500, f"Script not found: {script}")
    if not root_dir.exists():
        raise HTTPException(400, f"ROOT_DATASET_DIR not found: {root_dir}")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd_class(script, root_dir, out_dir, req.CLASS_TARGETS_JSON)
    gen = _stream_proc(cmd, cwd=script.parent, out_dir=out_dir, script_dir=script.parent)
    resp = StreamingResponse(gen, media_type="text/plain")
    resp.headers["X-Accel-Buffering"] = "no"
    return resp

# Optional: planning endpoint used by the UI to show availability/shortfall
class PlanRequest(BaseModel):
    ROOT_DATASET_DIR: str
    CLASS_TARGETS_JSON: str

@app.post("/augment/enhanced/class-specific/plan")
def plan_class_specific(req: PlanRequest):
    root_dir = _validate_path(req.ROOT_DATASET_DIR, "ROOT_DATASET_DIR")
    try:
        targets = json.loads(req.CLASS_TARGETS_JSON or "{}")
        if not isinstance(targets, dict):
            targets = {}
    except Exception:
        targets = {}
    # naive availability = current images per class (you can replace w/ a smarter estimator)
    per_class = {}
    for d in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        count = 0
        for ext in (".png", ".jpg", ".jpeg"):
            count += len(list(d.rglob(f"*{ext}")))
        tgt = int(targets.get(d.name, 0) or 0)
        avail = max(0, count * (count - 1) // 2)  # max unique pairs, rough upper bound
        shortfall = max(0, tgt - avail)
        per_class[d.name] = {"available": avail, "shortfall": shortfall}
    return {"per_class": per_class}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
