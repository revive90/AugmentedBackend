import os, shutil, pickle, time, json, argparse
from datetime import datetime
import numpy as np, psutil

# faiss keeps throwing an error
try:
    import faiss
except ImportError:
    try:
        import faiss_cpu as faiss
    except ImportError:
        import faiss_gpu as faiss

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

INDEX_OUTPUT_DIR = os.path.abspath("Faiss_Indexes")
RESULTS_OUTPUT_DIR = os.path.abspath("augmentation_results")

try:
    from feature_extractor import FeatureExtractor
except Exception:
    FeatureExtractor = None

try:
    from utils import fuse_images
except Exception:

    import cv2, numpy as np
    def fuse_images(p1, p2, outp):
        i1, i2 = cv2.imread(p1), cv2.imread(p2)
        if i1 is None or i2 is None:
            return
        if i1.shape != i2.shape:
            i2 = cv2.resize(i2, (i1.shape[1], i1.shape[0]))
        fused = np.zeros_like(i1)
        fused[:, ::2] = i1[:, ::2]
        fused[:, 1::2] = i2[:, 1::2]
        cv2.imwrite(outp, fused)

# ---- streaming helpers ----
def emit_event(**kwargs):
    """Emit a single-line JSON event that the frontend parses."""
    try:
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

# ---- pair selection for percentage mode ----
def top_pairs_by_similarity(embeddings, top_k):
    """
    Return up to top_k pairs by cosine similarity (descending).
    Deterministic order.
    """
    n = embeddings.shape[0]
    if n < 2 or top_k <= 0:
        return []

    norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims = norm @ norm.T
    iu = np.triu_indices(n, k=1)
    vals = sims[iu]
    order = np.argsort(vals)[::-1]
    a = iu[0][order]
    b = iu[1][order]
    # clip to top_k
    a = a[:top_k]
    b = b[:top_k]
    return list(zip(a, b))

# ---- main ----
def main(ROOT_DATASET_DIR, AUGMENTED_OUTPUT_DIR, AUGMENTATION_TARGET_PERCENTAGE):
    if FeatureExtractor is None:
        raise RuntimeError(
            "feature_extractor.FeatureExtractor is required but could not be imported."
        )

    overall_start = time.time()
    mem = MemoryTracker()
    emit_event(
        type="start",
        dataset_dir=ROOT_DATASET_DIR,
        output_dir=AUGMENTED_OUTPUT_DIR,
        mode="percentage",
        target_pct=AUGMENTATION_TARGET_PERCENTAGE,
    )

    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

    class_dirs = sorted([d.path for d in os.scandir(ROOT_DATASET_DIR) if d.is_dir()])
    classes = [os.path.basename(d) for d in class_dirs]
    if not classes:
        emit_event(type="done", elapsed_seconds=0, peak_mb=mem.get_peak())
        return

    # fresh outputs for this mode
    for path in (INDEX_OUTPUT_DIR, AUGMENTED_OUTPUT_DIR):
        try:
            if os.path.exists(path) and os.access(path, os.W_OK):
                shutil.rmtree(path)
        except Exception as e:
            print(f"[Warning] Could not delete {path}: {e}")
    os.makedirs(AUGMENTED_OUTPUT_DIR, exist_ok=True)

    # ---- Stage 1: Indexing ----
    stage1_start = time.time()
    fe = FeatureExtractor()
    per_class_stats = []

    for idx, class_dir in enumerate(class_dirs):
        class_name = os.path.basename(class_dir)
        all_paths = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        feats, paths = [], []
        for i, p in enumerate(all_paths):
            vec = fe.extract(p)
            if vec is not None:
                feats.append(vec)
                paths.append(p)
            if (i + 1) % 50 == 0 or (i + 1) == len(all_paths):
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
            class_index_dir = os.path.join(INDEX_OUTPUT_DIR, class_name)
            os.makedirs(class_index_dir, exist_ok=True)
            faiss.write_index(index, os.path.join(class_index_dir, "class.index"))
            with open(os.path.join(class_index_dir, "index_to_path.pkl"), "wb") as f:
                pickle.dump(paths, f)

        emit_event(
            type="overall_progress",
            phase="index",
            percent=(idx + 1) / max(1, len(class_dirs)) * 40.0,
            cls=class_name,
        )

    stage1_dur = time.time() - stage1_start

    # ---- Stage 2: Augmentation (percentage mode) ----
    total_generated = 0
    stage2_start = time.time()
    slice_per_class = 60.0 / float(len(class_dirs) or 1)

    for idx, class_dir in enumerate(class_dirs):
        class_name = os.path.basename(class_dir)
        stats = {
            "class": class_name,
            "images_in_class": 0,
            "index_size": 0,
            "target_pct": float(AUGMENTATION_TARGET_PERCENTAGE),
            "planned_new": 0,
            "available_pairs": 0,
            "used_pairs": 0,
            "generation_time_seconds": 0.0,
            "output_dir": None,
        }

        index_file = os.path.join(INDEX_OUTPUT_DIR, class_name, "class.index")
        map_file = os.path.join(INDEX_OUTPUT_DIR, class_name, "index_to_path.pkl")
        if not (os.path.exists(index_file) and os.path.exists(map_file)):
            emit_event(
                type="overall_progress",
                phase="augment",
                percent=40.0 + (idx + 1) * slice_per_class,
                cls=class_name,
            )
            per_class_stats.append(stats)
            continue

        index = faiss.read_index(index_file)
        with open(map_file, "rb") as f:
            image_paths = pickle.load(f)
        n = len(image_paths)
        stats["images_in_class"] = n
        stats["index_size"] = int(index.ntotal)
        if n < 2:
            per_class_stats.append(stats)
            continue

        # how many new images to generate for this class?
        needed = int(np.floor(n * (AUGMENTATION_TARGET_PERCENTAGE / 100.0)))
        stats["planned_new"] = needed

        # reconstruct embeddings and choose top pairs
        embeds = index.reconstruct_n(0, index.ntotal)
        mem.track()

        # Max possible unique pairs in a class
        max_pairs = n * (n - 1) // 2
        stats["available_pairs"] = int(max_pairs)
        top_k = min(needed, max_pairs)
        pairs = top_pairs_by_similarity(embeds, top_k)

        class_out = os.path.join(AUGMENTED_OUTPUT_DIR, class_name)
        os.makedirs(class_out, exist_ok=True)
        stats["output_dir"] = os.path.abspath(class_out)

        class_start = time.time()
        for j, (a, b) in enumerate(pairs):
            p1, p2 = image_paths[a], image_paths[b]
            outp = os.path.join(class_out, f"fused_{class_name}_{j}.png")
            fuse_images(p1, p2, outp)
            total_generated += 1

            if (j + 1) % 50 == 0 or (j + 1) == len(pairs):
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
                    cls=class_name,
                )
                emit_event(type="generated", total_generated=total_generated)

        stats["used_pairs"] = len(pairs)
        stats["generation_time_seconds"] = float(time.time() - class_start)
        per_class_stats.append(stats)

    stage2_dur = time.time() - stage2_start
    duration = time.time() - overall_start

    # textfile report
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)
    txt = os.path.join(RESULTS_OUTPUT_DIR, f"augmentation_percentage_mode_{ts}.txt")
    js = os.path.join(RESULTS_OUTPUT_DIR, f"augmentation_percentage_mode_{ts}.json")

    with open(txt, "w") as f:
        f.write("=== Enhanced OCMRI: Percentage Mode Summary ===\n")
        f.write(f"Run Timestamp: {ts}\n")
        f.write(f"Target Percentage (per-class): {AUGMENTATION_TARGET_PERCENTAGE}%\n\n")
        f.write("--- Timings ---\n")
        f.write(
            f"Indexing Time: {stage1_dur:.2f}s\nAugmentation Time: {stage2_dur:.2f}s\nTotal: {duration:.2f}s\n\n"
        )
        f.write(f"Peak Memory Usage: {mem.get_peak():.2f} MB\n")
        f.write(f"Total New Images Generated: {total_generated}\n")
        f.write(f"Total Classes Processed: {len(classes)}\n\n")
        f.write("--- Per-Class Details ---\n")
        for cs in per_class_stats:
            f.write(f"\nClass: {cs['class']}\n")
            f.write(f"  Images in Class: {cs['images_in_class']}\n")
            f.write(f"  Index Size: {cs['index_size']}\n")
            f.write(f"  Planned New (count): {cs['planned_new']}\n")
            f.write(f"  Available Pairs: {cs['available_pairs']}\n")
            f.write(f"  Used Pairs: {cs['used_pairs']}\n")
            f.write(f"  Generation Time (s): {cs['generation_time_seconds']:.2f}\n")
            f.write(f"  Output Directory: {cs['output_dir']}\n")

    try:
        with open(js, "w") as jf:
            json.dump(
                {
                    "mode": "percentage",
                    "target_pct": float(AUGMENTATION_TARGET_PERCENTAGE),
                    "timings": {
                        "indexing_s": stage1_dur,
                        "augmentation_s": stage2_dur,
                        "total_s": duration,
                    },
                    "resources": {"peak_memory_mb": mem.get_peak()},
                    "overall": {
                        "total_generated": total_generated,
                        "total_classes": len(classes),
                    },
                    "per_class": per_class_stats,
                },
                jf,
                indent=2,
            )
    except Exception:
        pass

    print("\n--- Percentage Mode Complete ---")
    print(f"Results: {txt}\nJSON: {js}\nTotal: {duration:.2f}s")
    emit_event(type="overall_progress", percent=100.0)
    emit_event(
        type="done",
        elapsed_seconds=duration,
        peak_mb=mem.get_peak(),
        summary_path=txt,
        summary_json=js,
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dataset-dir", type=str, required=True)
    ap.add_argument("--augmented-output-dir", type=str, required=True)
    ap.add_argument("--augmentation-target-percentage", type=float, required=True)
    args = ap.parse_args()

    main(
        args.root_dataset_dir,
        args.augmented_output_dir,
        args.augmentation_target_percentage,
    )
