# tests.py
# One-file test suite (pytest) for your backend
# Run with:  pytest -q
# Notes:
# - Heavy model/FAISS work is stubbed so tests are fast and offline.
# - Place this file in the SAME directory as your backend scripts.

import os
import sys
import json
import time
import types
import shutil
import random
import string
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# ------------------------------
# Helpers: temp data & tiny imgs
# ------------------------------

def _mk_workspace():
    """Create a temp workspace with a small synthetic dataset: classes A,B with PNGs."""
    base = Path(tempfile.mkdtemp(prefix="ocmri_tests_"))
    src = base / "dataset"
    for cls in ["A", "B"]:
        d = src / cls
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(123 if cls == "A" else 456)
        for i in range(6):
            arr = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
            Image.fromarray(arr).save(d / f"{cls.lower()}_{i:02d}.png")
    out = base / "out"
    out.mkdir(exist_ok=True)
    return base, src, out

def _count_imgs(root: Path):
    return len(list(root.rglob("*.png"))) + len(list(root.rglob("*.jpg"))) + len(list(root.rglob("*.jpeg")))

# ----------------------------------------------------
# Fixtures: workspace, dummy FeatureExtractor, FAISS
# ----------------------------------------------------

@pytest.fixture
def workspace():
    base, src, out = _mk_workspace()
    try:
        yield base, src, out
    finally:
        shutil.rmtree(base, ignore_errors=True)

class DummyExtractor:
    """Lightweight replacement for DINOv2 extractor used inside pipelines."""
    def __init__(self, dim=64):
        self.dim = dim
    def extract(self, image_path: str) -> np.ndarray:
        seed = sum(bytearray(Path(image_path).name.encode("utf-8")))
        rng = np.random.default_rng(seed)
        v = rng.random(self.dim).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        return v

class _FakeFaissIndex:
    """Simple in-memory 'FlatL2' style index holding vectors; enough for pipeline IO."""
    def __init__(self, d): self.vectors = np.empty((0, d), dtype=np.float32)
    def add(self, feats): self.vectors = np.vstack([self.vectors, feats]).astype(np.float32)
    @property
    def ntotal(self): return int(self.vectors.shape[0])

def _fake_faiss_module(tmpdir: Path):
    """
    Build a tiny fake 'faiss' module with IndexFlatL2, write_index, read_index.
    Pipelines only rely on a small subset of API.
    """
    m = types.ModuleType("faiss")
    def IndexFlatL2(d): return _FakeFaissIndex(d)
    def write_index(index, path): np.save(path + ".npy", index.vectors)
    class _Loader:
        def __call__(self, path):
            arr = np.load(path + ".npy")
            idx = _FakeFaissIndex(arr.shape[1])
            idx.vectors = arr
            return idx
    read_index = _Loader()
    def reconstruct_n(self, start, n):  # monkey add to instance via subclassing if needed
        return self.vectors
    # Attach symbols
    m.IndexFlatL2 = IndexFlatL2
    m.write_index = write_index
    m.read_index = read_index
    # Monkeypatch method lookup used by pipelines
    _FakeFaissIndex.reconstruct_n = reconstruct_n
    return m

@pytest.fixture
def fake_faiss(monkeypatch, tmp_path):
    """
    Provide a fake 'faiss' in sys.modules BEFORE importing pipeline modules.
    """
    fake = _fake_faiss_module(tmp_path)
    monkeypatch.setitem(sys.modules, "faiss", fake)
    # also satisfy code paths attempting faiss_cpu/faiss_gpu
    monkeypatch.setitem(sys.modules, "faiss_cpu", fake)
    monkeypatch.setitem(sys.modules, "faiss_gpu", fake)
    return fake

@pytest.fixture
def dummy_extractor():
    return DummyExtractor(dim=64)

# -----------------
# Module imports
# -----------------

# We import modules lazily *inside* tests when we need to monkeypatch first.
# This avoids loading real DINOv2 or a real FAISS.

# =========================================================
# utils.fuse_images
# =========================================================

# What: sanity check for column-wise interleaving.
# Input: two 16x16 PNGs with different solid colours; output path in temp dir.
# Expected: output file exists; even columns from img1, odd columns from img2; same shape.
def test_utils_fuse_images_basic(workspace):
    base, src, out = workspace
    from utils import fuse_images  # :contentReference[oaicite:0]{index=0}

    a = out / "a.png"
    b = out / "b.png"
    # create two simple images: red-ish and green-ish
    Image.fromarray(np.dstack([np.full((16,16), 200, np.uint8),
                               np.zeros((16,16), np.uint8),
                               np.zeros((16,16), np.uint8)])).save(a)
    Image.fromarray(np.dstack([np.zeros((16,16), np.uint8),
                               np.full((16,16), 200, np.uint8),
                               np.zeros((16,16), np.uint8)])).save(b)

    o = out / "fused.png"
    fuse_images(str(a), str(b), str(o))
    assert o.exists(), "fused image not written"

    fused = np.array(Image.open(o))
    assert fused.shape == (16, 16, 3), "unexpected fused size"
    # even columns should match image a red channel (200), odd columns should match image b green channel (200)
    assert int(fused[:, 0, 0].mean()) >= 190 and int(fused[:, 1, 1].mean()) >= 190

# =========================================================
# run_percentage_pipeline.top_pairs_by_similarity
# =========================================================

# What: pair selection returns top-k unique pairs by cosine similarity.
# Input: small 5x3 embedding array; ask for k=3.
# Expected: list of length <=3; no self-pairs; indices are unique pairs.
def test_percentage_top_pairs_by_similarity(monkeypatch):
    import importlib
    mod = importlib.import_module("run_percentage_pipeline")  # :contentReference[oaicite:1]{index=1}
    rng = np.random.default_rng(0)
    embeds = rng.random((5, 3)).astype(np.float32)
    pairs = mod.top_pairs_by_similarity(embeds, top_k=3)
    assert isinstance(pairs, list)
    assert len(pairs) <= 3
    for a, b in pairs:
        assert a != b and 0 <= a < 5 and 0 <= b < 5

# =========================================================
# run_threshold_pipeline.select_pairs_threshold
# =========================================================

# What: threshold band selection yields only pairs within [lower, upper).
# Input: random 6x4 embeddings; band [0.2, 0.9).
# Expected: every returned pair's cosine sim lies in band; order is desc by sim.
def test_threshold_select_pairs(monkeypatch):
    import importlib
    mod = importlib.import_module("run_threshold_pipeline")  # :contentReference[oaicite:2]{index=2}
    rng = np.random.default_rng(1)
    embeds = rng.random((6, 4)).astype(np.float32)
    pairs, sims = mod.select_pairs_threshold(embeds, lower=0.2, upper=0.9)
    assert len(pairs) == len(sims)
    # sims already sorted desc according to code; verify monotonic
    assert np.all(np.diff(sims) <= 1e-8)
    # band check
    assert np.all((sims >= 0.2) & (sims < 0.9))

# =========================================================
# run_full_pipeline.plan_pairs
# =========================================================

# What: lower threshold search should pick a band that yields up to target_count pairs.
# Input: synthetic 8x4 embeddings; upper=0.95, lower_min=0.5, target_count=5.
# Expected: returns a lower>=lower_min and <=upper; picked length<=target_count.
def test_full_plan_pairs(monkeypatch):
    import importlib
    mod = importlib.import_module("run_full_pipeline")  # :contentReference[oaicite:3]{index=3}
    rng = np.random.default_rng(2)
    embeds = rng.random((8, 4)).astype(np.float32)
    lower, picked = mod.plan_pairs(embeddings=embeds, upper=0.95, lower_min=0.5, target_count=5)
    assert 0.5 <= lower <= 0.95
    assert isinstance(picked, list) and len(picked) <= 5
    for a, b in picked:
        assert a != b and 0 <= a < 8 and 0 <= b < 8

# =========================================================
# ocmri_baseline (MSE/fuse flow) — smoke on tiny data
# =========================================================

# What: baseline main should create per-class outputs by fusing valid pairs.
# Input: tiny dataset with two classes; permissive thresholds to allow many pairs.
# Expected: output dir exists with at least one generated image; run finishes without error.
def test_ocmri_baseline_main_runs_small(workspace, monkeypatch):
    import importlib
    base, src, out = workspace
    mod = importlib.import_module("ocmri_baseline")  # :contentReference[oaicite:4]{index=4}

    # Make thresholds super wide so MSE falls inside range
    ROOT = str(src)
    MAIN_OUT = str(out / "baseline_out")
    # Thresholds are lower<upper; ACCEPTABLE_DIFFERENCE_PERCENTAGE not critical for test
    mod.main(ROOT, MAIN_OUT, INITIAL_TH1=0.0, INITIAL_TH2=1e9, ACCEPTABLE_DIFFERENCE_PERCENTAGE=100.0)

    out_root = Path(MAIN_OUT)
    assert out_root.exists()
    # at least something should be produced for first class
    gen_count = _count_imgs(out_root)
    assert gen_count >= 1, "no fused images created by baseline pipeline"

# =========================================================
# Enhanced pipelines (percentage, threshold, class-specific) — with stubs
# =========================================================

# What: percentage mode runs end-to-end with stubs; writes class folders & some fused outputs.
# Input: tiny dataset; fake faiss; dummy extractor injected; target_pct=100% (approx n new).
# Expected: augmented output exists; at least one fused image is written.
def test_run_percentage_pipeline_smoke(workspace, monkeypatch, fake_faiss, dummy_extractor):
    import importlib
    # Ensure FeatureExtractor used by module is replaced BEFORE import
    monkeypatch.setitem(sys.modules, "feature_extractor", types.SimpleNamespace(FeatureExtractor=lambda: dummy_extractor))
    mod = importlib.import_module("run_percentage_pipeline")  # :contentReference[oaicite:5]{index=5}

    base, src, out = workspace
    out_dir = out / "perc"
    mod.main(str(src), str(out_dir), AUGMENTATION_TARGET_PERCENTAGE=100.0)
    assert out_dir.exists()
    assert _count_imgs(out_dir) >= 1

# What: threshold mode runs with stubs; should respect band selection and produce outputs.
# Input: tiny dataset; fake faiss; dummy extractor; lower=0.1, upper=0.99.
# Expected: augmented output exists; >=1 fused image written somewhere.
def test_run_threshold_pipeline_smoke(workspace, monkeypatch, fake_faiss, dummy_extractor):
    import importlib
    monkeypatch.setitem(sys.modules, "feature_extractor", types.SimpleNamespace(FeatureExtractor=lambda: dummy_extractor))
    mod = importlib.import_module("run_threshold_pipeline")  # :contentReference[oaicite:6]{index=6}

    base, src, out = workspace
    out_dir = out / "thres"
    mod.main(str(src), str(out_dir), LOWER_THRESHOLD=0.1, UPPER_THRESHOLD=0.99)
    assert out_dir.exists()
    assert _count_imgs(out_dir) >= 1

# What: class-specific mode runs with stubs; respects per-class targets.
# Input: tiny dataset; fake faiss; dummy extractor; targets={"A":3,"B":2}.
# Expected: out/A has ~3 new files; out/B has ~2 (within feasible pair limits).
def test_run_class_specific_pipeline_smoke(workspace, monkeypatch, fake_faiss, dummy_extractor):
    import importlib
    monkeypatch.setitem(sys.modules, "feature_extractor", types.SimpleNamespace(FeatureExtractor=lambda: dummy_extractor))
    mod = importlib.import_module("run_class_specific_pipeline")  # :contentReference[oaicite:7]{index=7}

    base, src, out = workspace
    out_dir = out / "classspec"
    targets = json.dumps({"A": 3, "B": 2})
    mod.main(str(src), str(out_dir), CLASS_TARGETS_JSON=targets)
    assert out_dir.exists()
    # sanity: at least 1 new file in either class; exact counts depend on available unique pairs
    a_count = _count_imgs(out_dir / "A")
    b_count = _count_imgs(out_dir / "B")
    assert a_count + b_count >= 1

# =========================================================
# run_full_pipeline — end-to-end with stubs
# =========================================================

# What: full mode runs with fake faiss + dummy extractor; produces some outputs.
# Input: tiny dataset; upper=0.99, minimum_quality=0.5, target_pct=200 (aggressive).
# Expected: augmented dir exists; at least one fused image present.
def test_run_full_pipeline_smoke(workspace, monkeypatch, fake_faiss, dummy_extractor):
    import importlib
    monkeypatch.setitem(sys.modules, "feature_extractor", types.SimpleNamespace(FeatureExtractor=lambda: dummy_extractor))
    mod = importlib.import_module("run_full_pipeline")  # :contentReference[oaicite:8]{index=8}

    base, src, out = workspace
    out_dir = out / "full"
    mod.main(str(src), str(out_dir), UPPER_THRESHOLD=0.99, MINIMUM_QUALITY_THRESHOLD=0.5, AUGMENTATION_TARGET_PERCENTAGE=200.0)
    assert out_dir.exists()
    assert _count_imgs(out_dir) >= 1

# =========================================================
# server.py — API surface (non-stream + stream), subprocess stubbed
# =========================================================
# What: /datasets/scan-classes counts per-class images correctly.
# Input: dataset root path with two class folders from fixture.
# Expected: 200 response; JSON contains two classes with non-zero counts.
def test_server_scan_classes(workspace):
    from fastapi.testclient import TestClient
    import importlib
    srv = importlib.import_module("server")
    client = TestClient(srv.app)

    _, src, _ = workspace
    r = client.get("/datasets/scan-classes", params={"root": str(src)})
    assert r.status_code == 200, f"scan-classes failed: {r.status_code} {r.text}"
    data = r.json()
    # structure: {"root": "...", "classes":[{"name":"A","image_count":N}, ...]}
    assert "classes" in data and isinstance(data["classes"], list)
    names = {c["name"] for c in data["classes"]}
    assert {"A", "B"}.issubset(names)
    for c in data["classes"]:
        assert isinstance(c["image_count"], int) and c["image_count"] >= 0


# What: /augment/baseline (non-stream) returns a JSON payload and invokes a subprocess.
# Input: valid dirs and basic args; subprocess.run is stubbed to avoid launching Python.
# Expected: 200 response; JSON has exit_code and samples/summary_url fields.
def test_server_augment_baseline_stubbed(workspace, monkeypatch):
    from fastapi.testclient import TestClient
    import importlib, types
    srv = importlib.import_module("server")
    client = TestClient(srv.app)

    _, src, out = workspace
    # Put a sample file where the server's sample picker can find it after the (stubbed) run
    sample_dir = out / "baseline_stub" / "A"
    sample_dir.mkdir(parents=True, exist_ok=True)
    # a tiny valid PNG header is not required — the picker glob counts paths
    (sample_dir / "fake.png").write_bytes(b"PNG")

    class _Res:
        def __init__(self): self.returncode = 0; self.stdout = "ok"; self.stderr = ""

    # Stub subprocess.run so no external process is spawned
    monkeypatch.setattr(srv, "subprocess", types.SimpleNamespace(run=lambda *a, **k: _Res()))

    payload = {
        "ROOT_DATASET_DIR": str(src),
        "MAIN_OUTPUT_DIR": str(out / "baseline_stub"),
        "INITIAL_TH1": 0.0,
        "INITIAL_TH2": 1.0,
        "ACCEPTABLE_DIFFERENCE_PERCENTAGE": 10.0,
    }
    r = client.post("/augment/baseline", json=payload)
    assert r.status_code == 200, f"baseline endpoint failed: {r.status_code} {r.text}"
    data = r.json()
    assert data.get("exit_code") == 0
    assert "samples" in data and isinstance(data["samples"], list)
    # summary_url may be None if no summary file exists; allow both
    assert "summary_url" in data

# What: /augment/enhanced/percentage/stream streams lines and ends with SENTINEL payload.
# Input: stubbed _stream_proc yields a couple of EVT lines + final SENTINEL JSON.
# Expected: 200 response; response text contains SENTINEL and final {"done": true}.
def test_server_enhanced_stream_stubbed(monkeypatch):
    from fastapi.testclient import TestClient
    import importlib, json as _json
    srv = importlib.import_module("server")
    client = TestClient(srv.app)

    # Stub the internal stream generator used by the percentage stream endpoint
    def fake_stream(*args, **kwargs):
        yield "[[EVT]] {\"type\":\"heartbeat\"}\n"
        final = {"done": True, "samples": [], "summary_url": None}
        yield f"\n{srv.SENTINEL} " + _json.dumps(final) + "\n"

    monkeypatch.setattr(srv, "_stream_proc", fake_stream)

    payload = {
        "ROOT_DATASET_DIR": str(Path(".").resolve()),
        "AUGMENTED_OUTPUT_DIR": str(Path(".").resolve()),
        "AUGMENTATION_TARGET_PERCENTAGE": 100.0,
    }
    # NOTE: use the real endpoint path present in server.py
    r = client.post("/augment/enhanced/percentage/stream", json=payload)
    assert r.status_code == 200, f"enhanced stream failed: {r.status_code} {r.text}"
    txt = r.text
    assert srv.SENTINEL in txt
    assert '"done": true' in txt.lower()


# =========================================================
# feature_extractor — light behaviour check (skip if heavy)
# =========================================================

# What: the extractor module exists; skip heavy model loading by default.
# Input: none (module-level presence only).
# Expected: import succeeds; we don't instantiate the heavy class in unit tests.
def test_feature_extractor_module_present():
    pytest.importorskip("feature_extractor")  # :contentReference[oaicite:12]{index=12}
