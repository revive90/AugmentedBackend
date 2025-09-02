import os
import shutil
import pickle
import time
import json
import argparse
from datetime import datetime

import numpy as np
import psutil

# faiss keeps throwing an error
try:
    import faiss
except ImportError:
    try:
        import faiss_cpu as faiss
    except ImportError:
        import faiss_gpu as faiss

#  OpenMP duplicate warning crashes in some environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Default folders (written relative to where the script runs)
INDEX_OUTPUT_DIR = os.path.abspath("Faiss_Indexes")
RESULTS_OUTPUT_DIR = os.path.abspath("augmentation_results")

try:
    from feature_extractor import FeatureExtractor  # must expose .extract(img_path)->np.ndarray
except Exception:
    FeatureExtractor = None

try:
    from utils import fuse_images
except Exception:
    import cv2

    def fuse_images(p1, p2, outp):
        """
        Simple fallback fusion: interleave columns from both images.
        """
        i1, i2 = cv2.imread(p1), cv2.imread(p2)
        if i1 is None or i2 is None:
            return
        if i1.shape != i2.shape:
            i2 = cv2.resize(i2, (i1.shape[1], i1.shape[0]))
        fused = np.zeros_like(i1)
        fused[:, ::2] = i1[:, ::2]
        fused[:, 1::2] = i2[:, 1::2]
        cv2.imwrite(outp, fused)

# -------------- event helpers --------------
def emit_event(**kwargs):
    """
    Print a single-line JSON envelope prefixed with [[EVT]] for the frontend to parse.
    """
    try:
        # make floats smaller to keep lines tidy
        for k, v in list(kwargs.items()):
            if isinstance(v, float):
                kwargs[k] = round(v, 4)
        print("[[EVT]] " + json.dumps(kwargs, ensure_ascii=True), flush=True)
    except Exception:
        pass


class MemoryTracker:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak = 0.0
        self.track()

    def track(self):
        cur = self.process.memory_info().rss / (1024 * 1024)
        if cur > self.peak:
            self.peak = cur

    def get_peak(self):
        return self.peak


# -------------- core helpers --------------
def _extract_class_images(root_dir):
    """
    Return (classes, paths_map) where:
      - classes: ordered list of class names
      - paths_map: {class_name: [abs_img_paths]}
    """
    class_dirs = sorted([d.path for d in os.scandir(root_dir) if d.is_dir()])
    classes = [os.path.basename(d) for d in class_dirs]
    paths_map = {}
    for class_dir, cls in zip(class_dirs, classes):
        paths = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        paths_map[cls] = paths
    return classes, paths_map


def _top_pairs_by_similarity(embeddings, target_count):
    """
    Deterministically choose up to 'target_count' highest-similaity DISTINCT pairs.
    Uses cosine similarity on L2-normalized embeddings; returns list of (i, j).
    """
    n = embeddings.shape[0]
    if n < 2 or target_count <= 0:
        return []

    # L2-normalize rows
    norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    # cosine similarity = dot product of normalized vectors
    sim = norm @ norm.T

    iu = np.triu_indices(n, k=1)
    sims = sim[iu]
    order = np.argsort(sims)[-1::-1]  # high->low
    a, b = iu[0][order], iu[1][order]
    pairs = list(zip(a, b))
    return pairs[: int(target_count)]


# -------------- main pipeline --------------
def main(ROOT_DATASET_DIR, AUGMENTED_OUTPUT_DIR, CLASS_TARGETS_JSON):
    if FeatureExtractor is None:
        raise RuntimeError(
            "feature_extractor.FeatureExtractor is required but could not be imported."
        )

    try:
        targets = json.loads(CLASS_TARGETS_JSON or "{}")
        if not isinstance(targets, dict):
            targets = {}
    except Exception:
        targets = {}

    overall_start = time.time()
    mem = MemoryTracker()
    emit_event(
        type="start",
        dataset_dir=ROOT_DATASET_DIR,
        output_dir=AUGMENTED_OUTPUT_DIR,
        mode="class-specific",
    )

    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

    # Collect classes + available images
    classes, paths_map = _extract_class_images(ROOT_DATASET_DIR)
    if not classes:
        emit_event(
            type="done",
            elapsed_seconds=0,
            peak_mb=mem.get_peak(),
            total_generated=0,
            summary_path=None,
            summary_json=None,
        )
        return

    # Fresh outputs (indexes and augmented)
    for path in (INDEX_OUTPUT_DIR, AUGMENTED_OUTPUT_DIR):
        try:
            if os.path.exists(path) and os.access(path, os.W_OK):
                shutil.rmtree(path)
        except Exception as e:
            print(f"[Warning] Could not delete {path}: {e}")
    os.makedirs(AUGMENTED_OUTPUT_DIR, exist_ok=True)

    # -------- Stage 1: Indexing (0% - 40%) --------
    stage1_start = time.time()
    fe = FeatureExtractor()
    index_info = {}

    total_classes = len(classes)
    for idx, cls in enumerate(classes):
        image_paths = paths_map.get(cls, [])
        feats, valid_paths = [], []

        for i, p in enumerate(image_paths):
            vec = fe.extract(p)
            if vec is not None:
                feats.append(vec)
                valid_paths.append(p)

            # heart beat every 50 or at end of class
            if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
                mem.track()
                emit_event(
                    type="heartbeat",
                    rss_mb=mem.process.memory_info().rss / (1024 * 1024),
                    peak_mb=mem.get_peak(),
                )

        if feats:
            feats = np.asarray(feats, dtype="float32")
            index = faiss.IndexFlatL2(feats.shape[1])
            index.add(feats)

            class_index_dir = os.path.join(INDEX_OUTPUT_DIR, cls)
            os.makedirs(class_index_dir, exist_ok=True)

            idx_path = os.path.join(class_index_dir, "class.index")
            map_path = os.path.join(class_index_dir, "index_to_path.pkl")
            faiss.write_index(index, idx_path)
            with open(map_path, "wb") as f:
                pickle.dump(valid_paths, f)

            index_info[cls] = {
                "index_file": idx_path,
                "map_file": map_path,
                "count": len(valid_paths),
            }
        else:
            index_info[cls] = {"index_file": None, "map_file": None, "count": 0}

        # progress to 40%
        done_fraction = (idx + 1) / max(1, total_classes)
        emit_event(
            type="overall_progress",
            phase="index",
            percent=done_fraction * 40.0,
            cls=cls,
        )

    stage1_dur = time.time() - stage1_start

    # -------- Stage 2: Augmentation by explicit class targets (40% - 100%) --------
    stage2_start = time.time()
    total_generated = 0
    per_class_stats = []

    if total_classes > 0:
        slice_per_class = 60.0 / float(total_classes)
    else:
        slice_per_class = 60.0

    for idx, cls in enumerate(classes):
        # Target count of *new* images for this class
        target = int(targets.get(cls, 0) or 0)
        stats = {
            "class": cls,
            "images_in_class": index_info[cls]["count"],
            "planned_pairs": int(target),
            "generated": 0,
            "generation_time_seconds": 0.0,
            "output_dir": None,
        }

        # If no work to do or no enough images to form pairs
        if (
            target <= 0
            or index_info[cls]["index_file"] is None
            or index_info[cls]["count"] < 2
        ):
            per_class_stats.append(stats)
            # still advance overall progress for this class
            emit_event(
                type="overall_progress",
                phase="augment",
                percent=40.0 + (idx + 1) * slice_per_class,
                cls=cls,
            )
            continue

        # Reconstruct features from Flat index (stores vectors)
        index = faiss.read_index(index_info[cls]["index_file"])
        with open(index_info[cls]["map_file"], "rb") as f:
            image_paths = pickle.load(f)

        embeds = index.reconstruct_n(0, index.ntotal)
        mem.track()

        # Cap target to maximum unique pairs n*(n-1)/2
        n = len(image_paths)
        max_pairs = n * (n - 1) // 2
        target_capped = min(target, max_pairs)

        # Choose top pairs (highest similarity first) deterministically
        pairs = _top_pairs_by_similarity(embeds, target_capped)

        # Output dir for new images of this class
        class_out = os.path.join(AUGMENTED_OUTPUT_DIR, cls)
        os.makedirs(class_out, exist_ok=True)
        stats["output_dir"] = os.path.abspath(class_out)

        cstart = time.time()
        for j, (a, b) in enumerate(pairs):
            p1, p2 = image_paths[a], image_paths[b]
            outp = os.path.join(class_out, f"fused_{cls}_{j}.png")
            fuse_images(p1, p2, outp)

            stats["generated"] += 1
            total_generated += 1

            # Heart beat occasionally
            if (j + 1) % 25 == 0 or (j + 1) == len(pairs):
                mem.track()
                emit_event(
                    type="heartbeat",
                    rss_mb=mem.process.memory_info().rss / (1024 * 1024),
                    peak_mb=mem.get_peak(),
                )
                class_progress = (j + 1) / float(len(pairs) or 1)
                overall_now = 40.0 + (idx * slice_per_class) + (
                    class_progress * slice_per_class
                )
                emit_event(
                    type="overall_progress",
                    phase="augment",
                    percent=overall_now,
                    cls=cls,
                )
                emit_event(type="generated", total_generated=int(total_generated))

        stats["generation_time_seconds"] = float(time.time() - cstart)
        per_class_stats.append(stats)

    stage2_dur = time.time() - stage2_start
    duration = time.time() - overall_start

    # -------- Reporting --------
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)
    txt = os.path.join(
        RESULTS_OUTPUT_DIR, f"augmentation_class_specific_mode_{ts}.txt"
    )
    js = os.path.join(
        RESULTS_OUTPUT_DIR, f"augmentation_class_specific_mode_{ts}.json"
    )

    # augmentation text summary
    with open(txt, "w", encoding="utf-8") as f:
        f.write("=== Enhanced OCMRI: Class-Specific Mode Summary ===\n")
        f.write(f"Run Timestamp: {ts}\n\n")
        f.write("--- Timings ---\n")
        f.write(
            f"Indexing Time: {stage1_dur:.2f}s\nAugmentation Time: {stage2_dur:.2f}s\nTotal: {duration:.2f}s\n\n"
        )
        f.write(f"Peak Memory Usage: {mem.get_peak():.2f} MB\n")
        f.write(f"Total New Images Generated: {total_generated}\n")
        f.write(f"Total Classes (any): {len(classes)}\n\n")
        f.write("--- Per-Class Details ---\n")
        for cs in per_class_stats:
            f.write(f"\nClass: {cs['class']}\n")
            f.write(f"  Images in Class: {cs['images_in_class']}\n")
            f.write(f"  Planned (target): {cs['planned_pairs']}\n")
            f.write(f"  Generated: {cs['generated']}\n")
            f.write(f"  Generation Time (s): {cs['generation_time_seconds']:.2f}\n")
            f.write(f"  Output Directory: {cs['output_dir']}\n")

    # JSON
    try:
        with open(js, "w", encoding="utf-8") as jf:
            json.dump(
                {
                    "mode": "class-specific",
                    "timings": {
                        "indexing_s": stage1_dur,
                        "augmentation_s": stage2_dur,
                        "total_s": duration,
                    },
                    "resources": {"peak_memory_mb": mem.get_peak()},
                    "overall": {
                        "total_generated": int(total_generated),
                        "total_classes": len(classes),
                    },
                    "per_class": per_class_stats,
                },
                jf,
                indent=2,
            )
    except Exception:
        pass

    # Final console + events
    print("\n--- Class-Specific Mode Complete ---")
    print(f"Results: {txt}\nJSON: {js}\nTotal: {duration:.2f}s")

    # Let the server latch onto the exact summary + count
    emit_event(type="overall_progress", percent=100.0)
    emit_event(
        type="done",
        elapsed_seconds=duration,
        peak_mb=mem.get_peak(),
        total_generated=int(total_generated),
        summary_path=txt,
        summary_json=js,
    )


# -------------- CLI --------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dataset-dir", type=str, required=True)
    ap.add_argument("--augmented-output-dir", type=str, required=True)
    ap.add_argument(
        "--class-targets-json",
        type=str,
        required=True,
        help='JSON string like {"classA": 300, "classB": 120, ...}',
    )
    args = ap.parse_args()

    main(
        args.root_dataset_dir,
        args.augmented_output_dir,
        args.class_targets_json,
    )
