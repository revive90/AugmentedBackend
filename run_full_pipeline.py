"""Full planning pipeline for image pairing and augmentation.

Overview:
- End-to-end script for planning image pairs using feature extraction and similarity search.
- Incorporates configurable targets (e.g., augmentation percentage), quality thresholds, class targeting, and upper bounds to control selection.
- Builds or queries a similarity index (FAISS if available) and plans pairs that satisfy multiple constraints.
- Supports optional fused preview outputs and writes detailed results to designated output directories.
- This was the first iteration of the pipeline, before separating out the two stages, and the modes of operation..
- This script is still used by the existing pipeline, but is now deprecated.
Key components:
- fuse_images: Helper to produce combined visualizations of selected pairs.
- emit_event: Event reporting mechanism for progress and metrics.
- plan_pairs: Core planner balancing class targets, minimum quality thresholds, and upper caps.
- main: CLI driver that loads data, extracts features, plans pairs, and exports artifacts.

Intended use:
- Run as a script to create a balanced, constraint-aware pairing plan suitable for dataset augmentation.
"""


import os, shutil, pickle, time, json, argparse
from datetime import datetime
import numpy as np
import psutil

                                                                                                     
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

                                                                  
    stage1_start = time.time()
    per_class_stats = []                 
    config_snapshot = {
        "dataset_dir": os.path.abspath(ROOT_DATASET_DIR),
        "augmented_output_dir": os.path.abspath(AUGMENTED_OUTPUT_DIR),
        "index_output_dir": os.path.abspath(INDEX_OUTPUT_DIR),
        "results_output_dir": os.path.abspath(RESULTS_OUTPUT_DIR),
        "upper_threshold": UPPER_THRESHOLD,
        "minimum_quality_threshold": MINIMUM_QUALITY_THRESHOLD,
        "augmentation_target_percentage": AUGMENTATION_TARGET_PERCENTAGE,
        "class_targets_overrides": {k: v for k, v in CLASS_TARGETS.items()},
        "classes_found": classes,
        "start_time_iso": datetime.now().isoformat(timespec="seconds")
    }

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

    stage1_end = time.time()
    stage1_duration = stage1_end - stage1_start

    print("\n--- STAGE 2: Starting Online Augmentation ---")
    os.makedirs(AUGMENTED_OUTPUT_DIR, exist_ok=True)

    total_generated = 0
    slice_per_class = 60.0 / float(len(class_dirs))
    stage2_start = time.time()

    for idx, class_dir in enumerate(class_dirs):
        class_name = os.path.basename(class_dir)
        class_stats = {
            "class": class_name,
            "images_in_class": 0,
            "index_size": 0,
            "target_count": 0,
            "planned_pairs": 0,
            "lower_threshold": None,
            "upper_threshold": UPPER_THRESHOLD,
            "available_pairs_at_chosen_band": 0,
            "available_pairs_at_min_band": 0,
            "shortfall": 0,
            "sim_selected_min": None,
            "sim_selected_max": None,
            "sim_selected_avg": None,
            "generation_time_seconds": 0.0,
            "output_dir": None
        }

        print(f"\nAugmenting class: {class_name}")
        class_start = time.time()

        index_file = os.path.join(INDEX_OUTPUT_DIR, class_name, "class.index")
        map_file = os.path.join(INDEX_OUTPUT_DIR, class_name, "index_to_path.pkl")
        if not (os.path.exists(index_file) and os.path.exists(map_file)):
            emit_event(type="overall_progress", percent=40.0 + (idx + 1) * slice_per_class, phase="augment", cls=class_name)
            per_class_stats.append(class_stats)
            continue

        index = faiss.read_index(index_file)
        with open(map_file, "rb") as f:
            image_paths = pickle.load(f)

        class_stats["images_in_class"] = len(image_paths)
        class_stats["index_size"] = int(index.ntotal)

        target_count = CLASS_TARGETS.get(class_name, int(len(image_paths) * (AUGMENTATION_TARGET_PERCENTAGE / 100.0)))
        class_stats["target_count"] = int(target_count)

        if index.ntotal == 0 or target_count <= 0:
            emit_event(type="overall_progress", percent=40.0 + (idx + 1) * slice_per_class, phase="augment", cls=class_name)
            per_class_stats.append(class_stats)
            continue

        embeds = index.reconstruct_n(0, index.ntotal)
        mem.track()

        lower_th, pairs = plan_pairs(embeddings=embeds, upper=UPPER_THRESHOLD, lower_min=MINIMUM_QUALITY_THRESHOLD, target_count=target_count)
        class_stats["lower_threshold"] = float(lower_th)
        class_stats["planned_pairs"] = int(len(pairs))

        print(f"  Planned {len(pairs)} pairs with lower={lower_th:.2f}, upper={UPPER_THRESHOLD:.2f}")

                                                                            
        try:
            norm = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
            sim_mat = norm @ norm.T
                                                             
            n = norm.shape[0]
            iu = np.triu_indices(n, k=1)
            sims_all = sim_mat[iu]
                         
            mask_chosen = (sims_all >= lower_th) & (sims_all < UPPER_THRESHOLD)
            class_stats["available_pairs_at_chosen_band"] = int(np.count_nonzero(mask_chosen))
                                         
            mask_min = (sims_all >= MINIMUM_QUALITY_THRESHOLD) & (sims_all < UPPER_THRESHOLD)
            class_stats["available_pairs_at_min_band"] = int(np.count_nonzero(mask_min))
                           
            if pairs:
                sel_sims = np.array([sim_mat[a, b] for (a, b) in pairs], dtype=np.float32)
                class_stats["sim_selected_min"] = float(np.min(sel_sims))
                class_stats["sim_selected_max"] = float(np.max(sel_sims))
                class_stats["sim_selected_avg"] = float(np.mean(sel_sims))
        except Exception as _e:
                                                                                      
            pass

        if not pairs:
            emit_event(type="overall_progress", percent=40.0 + (idx + 1) * slice_per_class, phase="augment", cls=class_name)
            class_stats["shortfall"] = int(max(0, target_count))
            per_class_stats.append(class_stats)
            continue

        class_out = os.path.join(AUGMENTED_OUTPUT_DIR, class_name)
        os.makedirs(class_out, exist_ok=True)
        class_stats["output_dir"] = os.path.abspath(class_out)

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

                                
        class_end = time.time()
        class_stats["generation_time_seconds"] = float(class_end - class_start)
        class_stats["shortfall"] = int(max(0, target_count - len(pairs)))
        per_class_stats.append(class_stats)

    stage2_end = time.time()
    stage2_duration = stage2_end - stage2_start

    overall_end = time.time()
    duration = overall_end - overall_start

                                                                
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"augmentation_newPipeline_results_{ts}.txt"
    json_filename = f"augmentation_newPipeline_results_{ts}.json"
    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_OUTPUT_DIR, log_filename)
    json_path = os.path.join(RESULTS_OUTPUT_DIR, json_filename)

                 
    with open(log_path, "w") as f:
        f.write("=== Enhanced OCMRI Augmentation Summary ===\n")
        f.write(f"Run Timestamp: {ts}\n")
        f.write(f"Dataset Dir: {config_snapshot['dataset_dir']}\n")
        f.write(f"Augmented Output Dir: {config_snapshot['augmented_output_dir']}\n")
        f.write(f"Index Output Dir: {config_snapshot['index_output_dir']}\n")
        f.write(f"Results Output Dir: {config_snapshot['results_output_dir']}\n\n")

        f.write("--- Configuration ---\n")
        f.write(f"Upper Threshold (U): {UPPER_THRESHOLD}\n")
        f.write(f"Minimum Quality Threshold (L_min): {MINIMUM_QUALITY_THRESHOLD}\n")
        f.write(f"Augmentation Target (% of class): {AUGMENTATION_TARGET_PERCENTAGE}\n")
        f.write(f"Class Target Overrides: {json.dumps(config_snapshot['class_targets_overrides'])}\n\n")

        f.write("--- Timings ---\n")
        f.write(f"Stage 1 (Indexing) Time: {stage1_duration:.2f} seconds\n")
        f.write(f"Stage 2 (Augmentation) Time: {stage2_duration:.2f} seconds\n")
        f.write(f"Total Execution Time: {duration:.2f} seconds\n\n")

        f.write("--- Resources ---\n")
        f.write(f"Peak Memory Usage: {mem.get_peak():.2f} MB\n\n")

        f.write("--- Overall Output ---\n")
        f.write(f"Total New Images Generated: {total_generated}\n")
        f.write(f"Total Classes Processed: {len(classes)}\n\n")

        f.write("--- Per-Class Details ---\n")
        for cs in per_class_stats:
            f.write(f"\nClass: {cs['class']}\n")
            f.write(f"  Images in Class: {cs['images_in_class']}\n")
            f.write(f"  Index Size: {cs['index_size']}\n")
            f.write(f"  Target Augmentations: {cs['target_count']}\n")
            f.write(f"  Planned Pairs: {cs['planned_pairs']}\n")
            f.write(f"  Thresholds Used: lower={cs['lower_threshold']}, upper={cs['upper_threshold']}\n")
            f.write(f"  Available Pairs at Chosen Band [lower, upper): {cs['available_pairs_at_chosen_band']}\n")
            f.write(f"  Available Pairs at Min Band   [L_min, upper): {cs['available_pairs_at_min_band']}\n")
            f.write(f"  Shortfall vs Target: {cs['shortfall']}\n")
            f.write(f"  Selected Similarity (min/avg/max): {cs['sim_selected_min']}/{cs['sim_selected_avg']}/{cs['sim_selected_max']}\n")
            f.write(f"  Generation Time (s): {cs['generation_time_seconds']:.2f}\n")
            f.write(f"  Output Directory: {cs['output_dir']}\n")

                                    
    summary_json = {
        "run": config_snapshot,
        "timings": {
            "stage1_indexing_seconds": stage1_duration,
            "stage2_augmentation_seconds": stage2_duration,
            "total_seconds": duration
        },
        "resources": {
            "peak_memory_mb": mem.get_peak()
        },
        "overall": {
            "total_generated": total_generated,
            "total_classes": len(classes)
        },
        "per_class": per_class_stats
    }
    try:
        with open(json_path, "w") as jf:
            json.dump(summary_json, jf, indent=2)
    except Exception:
        pass

    print("\n\n--- Full Pipeline Complete ---")
    print(f"--- Results saved to: '{log_path}' ---")
    print(f"--- JSON summary saved to: '{json_path}' ---")
    print(f"--- Total execution time: {duration:.2f} seconds. ---")

    emit_event(type="overall_progress", percent=100.0)
    emit_event(type="done", elapsed_seconds=duration, peak_mb=mem.get_peak(), summary_path=log_path, summary_json=json_path)

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
"""Full planning pipeline for image pairing and augmentation.

Overview:
- End-to-end orchestration for planning image pairs using feature extraction and similarity search.
- Incorporates configurable targets (e.g., augmentation percentage), quality thresholds, class targeting, and upper bounds to control selection.
- Builds or queries a similarity index (FAISS if available) and plans pairs that satisfy multiple constraints.
- Supports optional fused preview outputs and writes detailed results to designated output directories.

Key components:
- fuse_images: Helper to produce combined visualizations of selected pairs.
- emit_event: Event reporting mechanism for progress and metrics.
- plan_pairs: Core planner balancing class targets, minimum quality thresholds, and upper caps.
- main: CLI driver that loads data, extracts features, plans pairs, and exports artifacts.

Intended use:
- Run as a script to create a balanced, constraint-aware pairing plan suitable for dataset augmentation or curation workflows.
"""