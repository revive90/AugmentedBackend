import os, shutil, pickle, time, json, argparse
from datetime import datetime
import numpy as np
import psutil

# ---- FAISS import (robust) ----
try:
    import faiss
except ImportError:
    try:
        import faiss_cpu as faiss
    except ImportError:
        import faiss_gpu as faiss

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#///
# -------------------- Defaults --------------------
#ROOT_DATASET_DIR = os.path.abspath("../../data/organised_data/Dataset_B/Training")
#AUGMENTED_OUTPUT_DIR = os.path.abspath("../../data/augmented_data/Dino/Dataset_B/Training")
INDEX_OUTPUT_DIR = os.path.abspath("Faiss_Indexes")
RESULTS_OUTPUT_DIR = os.path.abspath("augmentation_results")

AUGMENTATION_TARGET_PERCENTAGE = 600
UPPER_THRESHOLD = 0.99
MINIMUM_QUALITY_THRESHOLD = 0.80

CLASS_TARGETS = {
    "glioma_tumor": 20441, "glioma": 20441,
    "meningioma_tumor": 11814, "meningioma": 11814,
    "no_tumor": 19185, "notumor": 19185,
    "pituitary_tumor": 13094, "pituitary": 13094
}

try:
    from feature_extractor import FeatureExtractor
except Exception:
    FeatureExtractor = None

try:
    from utils import fuse_images
except Exception:
    import cv2
    def fuse_images(p1, p2, outp):
        i1, i2 = cv2.imread(p1), cv2.imread(p2)
        if i1 is None or i2 is None: return
        if i1.shape != i2.shape:
            i2 = cv2.resize(i2, (i1.shape[1], i1.shape[0]))
        fused = np.zeros_like(i1)
        for c in range(i1.shape[1]):
            fused[:, c] = i1[:, c] if (c % 2 == 0) else i2[:, c]
        cv2.imwrite(outp, fused)

# -------------------- Streaming helpers --------------------
def emit_event(**kwargs):
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
        self.peak_memory_mb = 0.0
        self.track()
    def track(self):
        cur = self.process.memory_info().rss / (1024 * 1024)
        if cur > self.peak_memory_mb:
            self.peak_memory_mb = cur
    def get_peak(self):
        return self.peak_memory_mb

# -------------------- Pair selection logic --------------------
def plan_pairs(embeddings, upper, lower_min, target_count):
    n = embeddings.shape[0]
    if n < 2: return lower_min, []

    norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim = norm @ norm.T

    iu = np.triu_indices(n, k=1)
    sims = sim[iu]
    order = np.argsort(sims)[::-1]
    pairs_sorted = list(zip(iu[0][order], iu[1][order]))
    sims_sorted = sims[order]

    best_lower = lower_min
    picked = []
    for th in np.arange(upper - 0.01, lower_min - 0.01, -0.01):
        mask = (sims_sorted >= th) & (sims_sorted < upper)
        idxs = np.nonzero(mask)[0]
        if idxs.size >= target_count:
            best_lower = round(float(th), 2)
            picked = [pairs_sorted[i] for i in idxs[:target_count]]
            break

    if not picked:
        mask = (sims_sorted >= lower_min) & (sims_sorted < upper)
        idxs = np.nonzero(mask)[0]
        picked = [pairs_sorted[i] for i in idxs]

    return best_lower, picked

# -------------------- Main pipeline --------------------
def main(ROOT_DATASET_DIR, AUGMENTED_OUTPUT_DIR, UPPER_THRESHOLD, MINIMUM_QUALITY_THRESHOLD, AUGMENTATION_TARGET_PERCENTAGE):
    if FeatureExtractor is None:
        raise RuntimeError("feature_extractor.FeatureExtractor is required but could not be imported.")

    overall_start = time.time()
    mem = MemoryTracker()
    emit_event(type="start", dataset_dir=ROOT_DATASET_DIR, output_dir=AUGMENTED_OUTPUT_DIR)

    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

    class_dirs = sorted([d.path for d in os.scandir(ROOT_DATASET_DIR) if d.is_dir()])
    classes = [os.path.basename(d) for d in class_dirs]
    print(f"Classes Found: {classes}")
    if not classes:
        emit_event(type="done", elapsed_seconds=0, peak_mb=mem.get_peak())
        return

    try:
        if os.path.exists(INDEX_OUTPUT_DIR) and os.access(INDEX_OUTPUT_DIR, os.W_OK):
            shutil.rmtree(INDEX_OUTPUT_DIR)
    except Exception as e:
        print(f"[Warning] Could not delete index dir: {INDEX_OUTPUT_DIR}. Reason: {e}")

    try:
        if os.path.exists(AUGMENTED_OUTPUT_DIR) and os.access(AUGMENTED_OUTPUT_DIR, os.W_OK):
            shutil.rmtree(AUGMENTED_OUTPUT_DIR)
    except Exception as e:
        print(f"[Warning] Could not delete augmented dir: {AUGMENTED_OUTPUT_DIR}. Reason: {e}")

    print("\n--- STAGE 1: Starting Offline Indexing ---")
    fe = FeatureExtractor()
    mem.track()

    for idx, class_dir in enumerate(class_dirs):
        class_name = os.path.basename(class_dir)
        print(f"\nIndexing class: {class_name}")

        all_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not all_paths:
            emit_event(type="overall_progress", percent=(idx + 1) / len(class_dirs) * 40.0, phase="index", cls=class_name)
            continue

        feats, paths = [], []
        for i, p in enumerate(all_paths):
            vec = fe.extract(p)
            if vec is not None:
                feats.append(vec)
                paths.append(p)
            if (i + 1) % 50 == 0 or (i + 1) == len(all_paths):
                mem.track()
                emit_event(type="heartbeat", rss_mb=mem.process.memory_info().rss / (1024 * 1024), peak_mb=mem.get_peak())

        if feats:
            feats_np = np.asarray(feats, dtype="float32")
            d = feats_np.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(feats_np)

            class_index_dir = os.path.join(INDEX_OUTPUT_DIR, class_name)
            os.makedirs(class_index_dir, exist_ok=True)
            faiss.write_index(index, os.path.join(class_index_dir, "class.index"))
            with open(os.path.join(class_index_dir, "index_to_path.pkl"), "wb") as f:
                pickle.dump(paths, f)

        emit_event(type="overall_progress", percent=(idx + 1) / len(class_dirs) * 40.0, phase="index", cls=class_name)

    print("\n--- STAGE 2: Starting Online Augmentation ---")
    os.makedirs(AUGMENTED_OUTPUT_DIR, exist_ok=True)

    total_generated = 0
    slice_per_class = 60.0 / float(len(class_dirs))

    for idx, class_dir in enumerate(class_dirs):
        class_name = os.path.basename(class_dir)
        print(f"\nAugmenting class: {class_name}")

        index_file = os.path.join(INDEX_OUTPUT_DIR, class_name, "class.index")
        map_file = os.path.join(INDEX_OUTPUT_DIR, class_name, "index_to_path.pkl")
        if not (os.path.exists(index_file) and os.path.exists(map_file)):
            emit_event(type="overall_progress", percent=40.0 + (idx + 1) * slice_per_class, phase="augment", cls=class_name)
            continue

        index = faiss.read_index(index_file)
        with open(map_file, "rb") as f:
            image_paths = pickle.load(f)

        target_count = CLASS_TARGETS.get(class_name, int(len(image_paths) * (AUGMENTATION_TARGET_PERCENTAGE / 100.0)))

        if index.ntotal == 0 or target_count <= 0:
            emit_event(type="overall_progress", percent=40.0 + (idx + 1) * slice_per_class, phase="augment", cls=class_name)
            continue

        embeds = index.reconstruct_n(0, index.ntotal)
        mem.track()

        lower_th, pairs = plan_pairs(embeddings=embeds, upper=UPPER_THRESHOLD, lower_min=MINIMUM_QUALITY_THRESHOLD, target_count=target_count)
        print(f"  Planned {len(pairs)} pairs with lower={lower_th:.2f}, upper={UPPER_THRESHOLD:.2f}")

        if not pairs:
            emit_event(type="overall_progress", percent=40.0 + (idx + 1) * slice_per_class, phase="augment", cls=class_name)
            continue

        class_out = os.path.join(AUGMENTED_OUTPUT_DIR, class_name)
        os.makedirs(class_out, exist_ok=True)

        for j, (a, b) in enumerate(pairs):
            p1, p2 = image_paths[a], image_paths[b]
            outp = os.path.join(class_out, f"fused_{class_name}_{j}.png")
            fuse_images(p1, p2, outp)
            total_generated += 1
            emit_event(type="generated", cls=class_name, generated_in_class=j + 1, total_generated=total_generated)

            if (j + 1) % 50 == 0 or (j + 1) == len(pairs):
                mem.track()
                emit_event(type="heartbeat", rss_mb=mem.process.memory_info().rss / (1024 * 1024), peak_mb=mem.get_peak())
                class_progress = (j + 1) / float(len(pairs))
                overall_now = 40.0 + (idx * slice_per_class) + (class_progress * slice_per_class)
                emit_event(type="overall_progress", percent=overall_now, phase="augment", cls=class_name)

    overall_end = time.time()
    duration = overall_end - overall_start

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"augmentation_newPipeline_results_{ts}.txt"
    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_OUTPUT_DIR, log_filename)
    with open(log_path, "w") as f:
        f.write(f"Total New Images Generated: {total_generated}\n")
        f.write(f"Peak Memory Usage: {mem.get_peak():.2f} MB\n")
        f.write(f"Total Execution Time: {duration:.2f} seconds\n")

    print("\n\n--- Full Pipeline Complete ---")
    print(f"--- Results saved to: '{log_path}' ---")
    print(f"--- Total execution time: {duration:.2f} seconds. ---")

    emit_event(type="overall_progress", percent=100.0)
    emit_event(type="done", elapsed_seconds=duration, peak_mb=mem.get_peak(), summary_path=log_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root-dataset-dir", type=str, required=True)
    p.add_argument("--augmented-output-dir", type=str, required=True)

    p.add_argument("--upper-threshold", type=float, default=UPPER_THRESHOLD)
    p.add_argument("--minimum-quality-threshold", type=float, default=MINIMUM_QUALITY_THRESHOLD)
    p.add_argument("--augmentation-target-percentage", type=float, default=AUGMENTATION_TARGET_PERCENTAGE)
    args = p.parse_args()

    main(
        ROOT_DATASET_DIR=args.root_dataset_dir,
        AUGMENTED_OUTPUT_DIR=args.augmented_output_dir,
        UPPER_THRESHOLD=args.upper_threshold,
        MINIMUM_QUALITY_THRESHOLD=args.minimum_quality_threshold,
        AUGMENTATION_TARGET_PERCENTAGE=args.augmentation_target_percentage,
    )
