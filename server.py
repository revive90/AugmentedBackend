# server.py
import sys, os, json, subprocess
from pathlib import Path
from typing import List, Optional, Iterator

from fastapi import FastAPI, HTTPException
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

class BaselineRequest(BaseModel):
    ROOT_DATASET_DIR: str
    MAIN_OUTPUT_DIR: str
    INITIAL_TH1: float
    INITIAL_TH2: float
    ACCEPTABLE_DIFFERENCE_PERCENTAGE: float

class EnhancedRequest(BaseModel):
    ROOT_DATASET_DIR: str
    AUGMENTED_OUTPUT_DIR: str
    UPPER_THRESHOLD: float
    MINIMUM_QUALITY_THRESHOLD: float
    AUGMENTATION_TARGET_PERCENTAGE: float

def to_static_url(p: Path) -> Optional[str]:
    try:
        rel = p.resolve().relative_to(SERVE_ROOT)
        return f"/static/{rel.as_posix()}"
    except ValueError:
        return None

def _latest_summary_url(script_dir: Path) -> Optional[str]:
    results = script_dir / "augmentation_results"
    if not results.exists():
        return None
    candidates = list(results.glob("*_augmentation_summary.txt")) + list(results.glob("augmentation_newPipeline_results_*.txt"))
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
            urls.append(url); continue
        dest = previews_root / p.name
        try:
            if not dest.exists():
                dest.write_bytes(p.read_bytes())
            rel = dest.resolve().relative_to(SERVE_ROOT)
            urls.append(f"/static/{rel.as_posix()}")
        except Exception:
            pass
    return urls

# ---------- Safety check ----------
def _validate_path(path_str: str, label: str):
    if not path_str.strip():
        raise HTTPException(400, f"{label} must not be empty.")
    path = Path(path_str).resolve()
    if path == Path("/"):
        raise HTTPException(400, f"{label} path is invalid or too broad.")
    return path

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

SENTINEL = "[[__DONE__]]"

def _stream_proc(cmd: List[str], cwd: Path, out_dir: Path, script_dir: Path) -> Iterator[str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=str(cwd), encoding="utf-8", errors="replace")
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                yield line.replace("\r", "\n")
    finally:
        proc.wait()
    samples = pick_sample_images_from_classes(out_dir, limit=5)
    summary_url = _latest_summary_url(script_dir)
    yield f"\n{SENTINEL} " + json.dumps({"done": True, "samples": samples, "summary_url": summary_url}) + "\n"

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

# ----------------- Enhanced endpoints -----------------
def _build_cmd_enhanced(script: Path, root_dir: Path, out_dir: Path, upper: float, lower: float, pct: float) -> List[str]:
    return [
        sys.executable, "-u", str(script),
        "--root-dataset-dir", str(root_dir),
        "--augmented-output-dir", str(out_dir),
        "--upper-threshold", str(upper),
        "--minimum-quality-threshold", str(lower),
        "--augmentation-target-percentage", str(pct),
    ]

@app.post("/augment/enhanced")
def run_enhanced(req: EnhancedRequest):
    script = (Path(__file__).parent / "run_full_pipeline.py").resolve()
    root_dir = _validate_path(req.ROOT_DATASET_DIR, "ROOT_DATASET_DIR")
    out_dir = _validate_path(req.AUGMENTED_OUTPUT_DIR, "AUGMENTED_OUTPUT_DIR")

    if not script.exists():
        raise HTTPException(500, f"Script not found: {script}")
    if not root_dir.exists():
        raise HTTPException(400, f"ROOT_DATASET_DIR not found: {root_dir}")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd_enhanced(script, root_dir, out_dir, req.UPPER_THRESHOLD, req.MINIMUM_QUALITY_THRESHOLD, req.AUGMENTATION_TARGET_PERCENTAGE)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(script.parent))

    samples = pick_sample_images_from_classes(out_dir, limit=5)
    summary_url = _latest_summary_url(script.parent)
    return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "samples": samples, "summary_url": summary_url}

@app.post("/augment/enhanced/stream")
def run_enhanced_stream(req: EnhancedRequest):
    script = (Path(__file__).parent / "run_full_pipeline.py").resolve()
    root_dir = _validate_path(req.ROOT_DATASET_DIR, "ROOT_DATASET_DIR")
    out_dir = _validate_path(req.AUGMENTED_OUTPUT_DIR, "AUGMENTED_OUTPUT_DIR")

    if not script.exists():
        raise HTTPException(500, f"Script not found: {script}")
    if not root_dir.exists():
        raise HTTPException(400, f"ROOT_DATASET_DIR not found: {root_dir}")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd_enhanced(script, root_dir, out_dir, req.UPPER_THRESHOLD, req.MINIMUM_QUALITY_THRESHOLD, req.AUGMENTATION_TARGET_PERCENTAGE)
    gen = _stream_proc(cmd, cwd=script.parent, out_dir=out_dir, script_dir=script.parent)
    resp = StreamingResponse(gen, media_type="text/plain")
    resp.headers["X-Accel-Buffering"] = "no"
    return resp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
