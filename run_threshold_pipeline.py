import os, shutil, pickle, time, json, argparse
from datetime import datetime
import numpy as np, psutil

try:
    import faiss
except ImportError:
    try: import faiss_cpu as faiss
    except ImportError: import faiss_gpu as faiss

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
        if i1 is None or i2 is None: return
        if i1.shape != i2.shape: i2 = cv2.resize(i2, (i1.shape[1], i1.shape[0]))
        fused = np.zeros_like(i1)
        for c in range(i1.shape[1]): fused[:, c] = i1[:, c] if (c % 2 == 0) else i2[:, c]
        cv2.imwrite(outp, fused)

def emit_event(**kwargs):
    try:
        for k, v in list(kwargs.items()):
            if isinstance(v, float): kwargs[k] = round(v, 4)
        print("[[EVT]] " + json.dumps(kwargs, ensure_ascii=True), flush=True)
    except Exception: pass

class MemoryTracker:
    def __init__(self): self.process = psutil.Process(os.getpid()); self.peak = 0.0; self.track()
    def track(self):
        cur = self.process.memory_info().rss/(1024*1024)
        if cur > self.peak: self.peak = cur
    def get_peak(self): return self.peak

def select_pairs_threshold(embeddings, lower, upper):
    """Return ALL pairs with similarity in [lower, upper). Deterministic order (desc)."""
    n = embeddings.shape[0]
    if n < 2: return []
    norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim = norm @ norm.T
    iu = np.triu_indices(n, k=1)
    sims = sim[iu]
    order = np.argsort(sims)[::-1]  # high to low (deterministic)
    a, b = iu[0][order], iu[1][order]
    mask = (sims[order] >= lower) & (sims[order] < upper)
    return list(zip(a[mask], b[mask])), sims[order][mask]

def main(ROOT_DATASET_DIR, AUGMENTED_OUTPUT_DIR, LOWER_THRESHOLD, UPPER_THRESHOLD):
    if FeatureExtractor is None:
        raise RuntimeError("feature_extractor.FeatureExtractor is required but could not be imported.")

    overall_start = time.time()
    mem = MemoryTracker()
    emit_event(type="start", dataset_dir=ROOT_DATASET_DIR, output_dir=AUGMENTED_OUTPUT_DIR)

    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

    class_dirs = sorted([d.path for d in os.scandir(ROOT_DATASET_DIR) if d.is_dir()])
    classes = [os.path.basename(d) for d in class_dirs]
    if not classes:
        emit_event(type="done", elapsed_seconds=0, peak_mb=mem.get_peak()); return

    # fresh outputs
    for path in (INDEX_OUTPUT_DIR, AUGMENTED_OUTPUT_DIR):
        try:
            if os.path.exists(path) and os.access(path, os.W_OK): shutil.rmtree(path)
        except Exception as e: print(f"[Warning] Could not delete {path}: {e}")
    os.makedirs(AUGMENTED_OUTPUT_DIR, exist_ok=True)

    # --- Stage 1: Indexing ---
    stage1_start = time.time()
    fe = FeatureExtractor()
    per_class_stats = []
    for idx, class_dir in enumerate(class_dirs):
        class_name = os.path.basename(class_dir)
        all_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                     if f.lower().endswith((".png",".jpg",".jpeg"))]
        feats, paths = [], []
        for i, p in enumerate(all_paths):
            vec = fe.extract(p)
            if vec is not None: feats.append(vec); paths.append(p)
            if (i+1)%50==0 or (i+1)==len(all_paths):
                mem.track()
                emit_event(type="heartbeat",
                           rss_mb=mem.process.memory_info().rss/(1024*1024),
                           peak_mb=mem.get_peak())
        if feats:
            feats = np.asarray(feats, dtype="float32")
            index = faiss.IndexFlatL2(feats.shape[1]); index.add(feats)
            class_index_dir = os.path.join(INDEX_OUTPUT_DIR, class_name)
            os.makedirs(class_index_dir, exist_ok=True)
            faiss.write_index(index, os.path.join(class_index_dir, "class.index"))
            with open(os.path.join(class_index_dir, "index_to_path.pkl"), "wb") as f:
                pickle.dump(paths, f)
        emit_event(type="overall_progress", phase="index",
                   percent=(idx+1)/max(1,len(class_dirs))*40.0, cls=class_name)
    stage1_dur = time.time() - stage1_start

    # --- Stage 2: Augmentation (threshold mode) ---
    total_generated = 0
    stage2_start = time.time()
    slice_per_class = 60.0/float(len(class_dirs) or 1)

    for idx, class_dir in enumerate(class_dirs):
        class_name = os.path.basename(class_dir)
        stats = {
            "class": class_name, "images_in_class": 0, "index_size": 0,
            "lower_threshold": float(LOWER_THRESHOLD), "upper_threshold": float(UPPER_THRESHOLD),
            "planned_pairs": 0, "available_pairs_in_band": 0,
            "sim_selected_min": None, "sim_selected_avg": None, "sim_selected_max": None,
            "generation_time_seconds": 0.0, "output_dir": None
        }
        index_file = os.path.join(INDEX_OUTPUT_DIR, class_name, "class.index")
        map_file   = os.path.join(INDEX_OUTPUT_DIR, class_name, "index_to_path.pkl")
        if not (os.path.exists(index_file) and os.path.exists(map_file)):
            emit_event(type="overall_progress", phase="augment",
                       percent=40.0+(idx+1)*slice_per_class, cls=class_name)
            per_class_stats.append(stats); continue

        index = faiss.read_index(index_file)
        with open(map_file, "rb") as f: image_paths = pickle.load(f)
        stats["images_in_class"] = len(image_paths); stats["index_size"] = int(index.ntotal)
        if index.ntotal < 2:
            per_class_stats.append(stats); continue

        embeds = index.reconstruct_n(0, index.ntotal); mem.track()
        class_start = time.time()
        pairs, sel_sims = select_pairs_threshold(embeds, LOWER_THRESHOLD, UPPER_THRESHOLD)
        stats["planned_pairs"] = len(pairs); stats["available_pairs_in_band"] = len(pairs)

        class_out = os.path.join(AUGMENTED_OUTPUT_DIR, class_name)
        os.makedirs(class_out, exist_ok=True); stats["output_dir"] = os.path.abspath(class_out)

        for j, (a, b) in enumerate(pairs):
            p1, p2 = image_paths[a], image_paths[b]
            outp = os.path.join(class_out, f"fused_{class_name}_{j}.png")
            fuse_images(p1, p2, outp); total_generated += 1
            if (j+1)%50==0 or (j+1)==len(pairs):
                mem.track()
                emit_event(type="heartbeat",
                           rss_mb=mem.process.memory_info().rss/(1024*1024),
                           peak_mb=mem.get_peak())
                class_progress = (j+1)/float(len(pairs) or 1)
                overall_now = 40.0 + (idx*slice_per_class) + (class_progress*slice_per_class)
                emit_event(type="overall_progress", phase="augment", percent=overall_now, cls=class_name)
                emit_event(
                    type="generated",
                    cls=class_name,
                    generated_in_class=j + 1,
                    total_generated=total_generated
                )

        if len(pairs)>0:
            stats["sim_selected_min"] = float(np.min(sel_sims))
            stats["sim_selected_avg"] = float(np.mean(sel_sims))
            stats["sim_selected_max"] = float(np.max(sel_sims))
        stats["generation_time_seconds"] = float(time.time()-class_start)
        per_class_stats.append(stats)

    stage2_dur = time.time() - stage2_start
    duration = time.time() - overall_start

    # --- Reporting ---
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)
    txt = os.path.join(RESULTS_OUTPUT_DIR, f"augmentation_threshold_mode_{ts}.txt")
    js  = os.path.join(RESULTS_OUTPUT_DIR, f"augmentation_threshold_mode_{ts}.json")

    with open(txt, "w") as f:
        f.write("=== Enhanced OCMRI: Threshold Mode Summary ===\n")
        f.write(f"Run Timestamp: {ts}\n")
        f.write(f"Lower Threshold (L): {LOWER_THRESHOLD}\nUpper Threshold (U): {UPPER_THRESHOLD}\n\n")
        f.write("--- Timings ---\n")
        f.write(f"Indexing Time: {stage1_dur:.2f}s\nAugmentation Time: {stage2_dur:.2f}s\nTotal: {duration:.2f}s\n\n")
        f.write(f"Peak Memory Usage: {mem.get_peak():.2f} MB\n")
        f.write(f"Total New Images Generated: {total_generated}\n")
        f.write(f"Total Classes Processed: {len(classes)}\n\n")
        f.write("--- Per-Class Details ---\n")
        for cs in per_class_stats:
            f.write(f"\nClass: {cs['class']}\n")
            f.write(f"  Images in Class: {cs['images_in_class']}\n")
            f.write(f"  Index Size: {cs['index_size']}\n")
            f.write(f"  Thresholds: lower={cs['lower_threshold']}, upper={cs['upper_threshold']}\n")
            f.write(f"  Available Pairs in Band: {cs['available_pairs_in_band']}\n")
            f.write(f"  Planned Pairs: {cs['planned_pairs']}\n")
            f.write(f"  Selected Similarity (min/avg/max): {cs['sim_selected_min']}/{cs['sim_selected_avg']}/{cs['sim_selected_max']}\n")
            f.write(f"  Generation Time (s): {cs['generation_time_seconds']:.2f}\n")
            f.write(f"  Output Directory: {cs['output_dir']}\n")

    try:
        with open(js, "w") as jf:
            json.dump({
                "mode": "threshold",
                "thresholds": {"lower": float(LOWER_THRESHOLD), "upper": float(UPPER_THRESHOLD)},
                "timings": {"indexing_s": stage1_dur, "augmentation_s": stage2_dur, "total_s": duration},
                "resources": {"peak_memory_mb": mem.get_peak()},
                "overall": {"total_generated": total_generated, "total_classes": len(classes)},
                "per_class": per_class_stats
            }, jf, indent=2)
    except Exception: pass

    print("\n--- Threshold Mode Complete ---")
    print(f"Results: {txt}\nJSON: {js}\nTotal: {duration:.2f}s")
    emit_event(type="overall_progress", percent=100.0)
    emit_event(type="done", elapsed_seconds=duration, peak_mb=mem.get_peak(), summary_path=txt, summary_json=js)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dataset-dir", type=str, required=True)
    ap.add_argument("--augmented-output-dir", type=str, required=True)
    ap.add_argument("--lower-threshold", type=float, required=True)
    ap.add_argument("--upper-threshold", type=float, required=True)
    args = ap.parse_args()
    main(args.root_dataset_dir, args.augmented_output_dir, args.lower_threshold, args.upper_threshold)
