# ocmri_baseline.py
import os
import shutil
import cv2
import numpy as np
import time
import sys
import argparse
import psutil
import json

# -------------------- Memory tracker --------------------
class MemoryTracker:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_memory_mb = 0
        self.track()

    def track(self):
        current_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        if current_memory_mb > self.peak_memory_mb:
            self.peak_memory_mb = current_memory_mb

    def get_peak_memory(self):
        return self.peak_memory_mb

# -------------------- Events + helpers --------------------
def emit_event(**kwargs):
    """Print a single-line JSON event for the UI; flushed immediately."""
    try:
        for k, v in list(kwargs.items()):
            if isinstance(v, float):
                kwargs[k] = round(v, 4)
        print("[[EVT]] " + json.dumps(kwargs, ensure_ascii=True), flush=True)
    except Exception:
        pass

def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='#', event=None):
    if total == 0:
        total = 1
    percent_val = 100 * (iteration / float(total))
    percent = f"{percent_val:.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if event:
        try:
            emit_event(
                type=event.get("type", "progress"),
                cls=event.get("cls"),
                phase=event.get("phase"),
                class_idx=event.get("class_idx"),
                num_classes=event.get("num_classes"),
                done=iteration,
                total=total,
                percent=percent_val
            )
        except Exception:
            pass
    if iteration == total:
        sys.stdout.write('\n')

def calculate_mse(image_path1, image_path2):
    try:
        img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            return float('inf')
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
        err /= float(img1.shape[0] * img1.shape[1])
        return err
    except Exception:
        return float('inf')

def fuse_images(image_path1, image_path2, output_path):
    img1, img2 = cv2.imread(image_path1), cv2.imread(image_path2)
    if img1 is None or img2 is None:
        return
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    fused_image = np.zeros_like(img1)
    for i in range(img1.shape[1]):
        if i % 2 == 0:
            fused_image[:, i] = img1[:, i]
        else:
            fused_image[:, i] = img2[:, i]
    cv2.imwrite(output_path, fused_image)

# -------------------- Compare pass with overall % --------------------
def get_fused_pairs_for_mse(
    class_dir,
    thresholds,
    memory_tracker,
    class_name,
    class_idx,
    num_classes,
    compare_peak=0.0,   # pass back in on repeated passes to avoid regressions
):
    image_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_paths) < 2:
        return [], compare_peak

    fused_pairs = []
    total_comparisons = (len(image_paths) * (len(image_paths) - 1)) // 2
    count = 0

    print_progress_bar(
        0, total_comparisons, prefix='  Comparing:', length=40,
        event={"type": "compare_progress", "cls": class_name, "phase": "compare",
               "class_idx": class_idx, "num_classes": num_classes}
    )
    # initial overall with current (peak) compare progress
    overall = ((class_idx * 100.0) + 0.5 * compare_peak + 0.0) / float(num_classes)
    emit_event(type="overall_progress", percent=clamp(overall), cls=class_name,
               class_idx=class_idx, num_classes=num_classes)

    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            count += 1

            if count % 200 == 0 or count == total_comparisons:
                print_progress_bar(
                    count, total_comparisons, prefix='  Comparing:', length=40,
                    event={"type": "compare_progress", "cls": class_name, "phase": "compare",
                           "class_idx": class_idx, "num_classes": num_classes}
                )
                memory_tracker.track()
                emit_event(
                    type="heartbeat",
                    cls=class_name,
                    rss_mb=memory_tracker.process.memory_info().rss / (1024 * 1024),
                    peak_mb=memory_tracker.get_peak_memory()
                )
                # update overall based on max compare so far
                current_percent = 100.0 * (count / float(total_comparisons))
                compare_peak = max(compare_peak, current_percent)
                overall = ((class_idx * 100.0) + 0.5 * compare_peak + 0.0) / float(num_classes)
                emit_event(type="overall_progress", percent=clamp(overall), cls=class_name,
                           class_idx=class_idx, num_classes=num_classes)

            path1, path2 = image_paths[i], image_paths[j]
            mse = calculate_mse(path1, path2)
            if thresholds['lower'] < mse < thresholds['upper']:
                fused_pairs.append((path1, path2))

    return fused_pairs, compare_peak

# -------------------- Main --------------------
def main(ROOT_DATASET_DIR, MAIN_OUTPUT_DIR, INITIAL_TH1, INITIAL_TH2, ACCEPTABLE_DIFFERENCE_PERCENTAGE):
    dataset_name = os.path.basename(ROOT_DATASET_DIR.rstrip("/\\")) or "Dataset_B"
    RESULTS_OUTPUT_DIR = "augmentation_results"

    CLASS_LIMITS = {
        "glioma_tumor": {'min': 500, 'max': 825},
        "glioma": {'min': 500, 'max': 825},
        "meningioma_tumor": {'min': 500, 'max': 1300},
        "meningioma": {'min': 500, 'max': 1300},
        "no_tumor": {'min': 500, 'max': 2000},
        "notumor": {'min': 500, 'max': 2000},
        "pituitary_tumor": {'min': 500, 'max': 1300},
        "pituitary": {'min': 500, 'max': 1300}
    }

    overall_start_time = time.time()
    memory_tracker = MemoryTracker()
    emit_event(type="start", dataset_dir=ROOT_DATASET_DIR, output_dir=MAIN_OUTPUT_DIR)

    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(ROOT_DATASET_DIR):
        print(f"Error: Dataset directory '{ROOT_DATASET_DIR}' not found.")
        return

    class_dirs = sorted([d.path for d in os.scandir(ROOT_DATASET_DIR) if d.is_dir()])
    num_classes = len(class_dirs)
    print(f"Found {num_classes} classes: {[os.path.basename(d) for d in class_dirs]}")

    if os.path.exists(MAIN_OUTPUT_DIR):
        shutil.rmtree(MAIN_OUTPUT_DIR)
    os.makedirs(MAIN_OUTPUT_DIR)
    print(f"Created main output directory: '{MAIN_OUTPUT_DIR}'")

    class_results = {}

    # ---------- First class ----------
    first_class_dir = class_dirs[0]
    class_name = os.path.basename(first_class_dir)
    class_idx = 0
    print(f"\n--- Processing first class to set baseline: {class_name} ---")

    initial_thresholds = {'lower': INITIAL_TH1, 'upper': INITIAL_TH2}
    compare_peak = 0.0
    fused_pairs_first_class, compare_peak = get_fused_pairs_for_mse(
        first_class_dir, initial_thresholds, memory_tracker,
        class_name, class_idx, num_classes, compare_peak
    )

    class_output_dir = os.path.join(MAIN_OUTPUT_DIR, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    total_to_fuse = len(fused_pairs_first_class)
    print_progress_bar(
        0, total_to_fuse, prefix='  Fusing:', length=40,
        event={"type": "fuse_progress", "cls": class_name, "phase": "fuse",
               "class_idx": class_idx, "num_classes": num_classes}
    )
    # Overall at start of fuse: compare contributes full 50%
    overall = ((class_idx * 100.0) + 50.0 + 0.0) / float(num_classes)
    emit_event(type="overall_progress", percent=clamp(overall), cls=class_name,
               class_idx=class_idx, num_classes=num_classes)

    generated_so_far = 0
    denom = max(1, total_to_fuse)
    for i, (p1, p2) in enumerate(fused_pairs_first_class):
        outp = os.path.join(class_output_dir, f"fused_{class_name}_{i}.png")
        fuse_images(p1, p2, outp)
        generated_so_far += 1
        print_progress_bar(
            i + 1, total_to_fuse, prefix='  Fusing:', length=40,
            event={"type": "fuse_progress", "cls": class_name, "phase": "fuse",
                   "class_idx": class_idx, "num_classes": num_classes}
        )
        fuse_percent = 100.0 * ((i + 1) / float(denom))
        overall = ((class_idx * 100.0) + 50.0 + 0.5 * fuse_percent) / float(num_classes)
        emit_event(type="overall_progress", percent=clamp(overall), cls=class_name,
                   class_idx=class_idx, num_classes=num_classes)
        emit_event(type="generated", cls=class_name, generated_so_far=generated_so_far)

    num_original_first = len(os.listdir(first_class_dir))
    num_fused_first = len(fused_pairs_first_class)
    target_image_count = num_original_first + num_fused_first

    class_results[class_name] = {'original': num_original_first, 'generated': num_fused_first}
    print(f"Generated {num_fused_first} images for '{class_name}'.")
    print(f"Target image count for all other classes is now: {target_image_count}")
    emit_event(type="class_done", cls=class_name,
               original=num_original_first, generated=num_fused_first,
               total_after=target_image_count)

    # ---------- Subsequent classes ----------
    for class_idx in range(1, num_classes):
        current_class_dir = class_dirs[class_idx]
        class_name = os.path.basename(current_class_dir)
        print(f"\n--- Dynamically processing class: {class_name} ---")

        num_original_current = len(os.listdir(current_class_dir))

        th2 = INITIAL_TH2
        min_th2 = CLASS_LIMITS.get(class_name, {}).get('min', INITIAL_TH1)
        max_th2 = CLASS_LIMITS.get(class_name, {}).get('max', INITIAL_TH2 * 2)
        print(f"  (Constraining Th2 search for this class between {min_th2} and {max_th2})")

        final_pairs_for_this_class = []
        compare_peak = 0.0  # maintain peak across multiple compare passes for this class
        while True:
            print(f"  Adjusting Th2 (current value: {th2:.2f})...")
            current_thresholds = {'lower': INITIAL_TH1, 'upper': th2}
            fused_pairs_current_class, compare_peak = get_fused_pairs_for_mse(
                current_class_dir, current_thresholds, memory_tracker,
                class_name, class_idx, num_classes, compare_peak
            )
            total_current_class = num_original_current + len(fused_pairs_current_class)

            diff_percentage = 100 if target_image_count == 0 else abs(
                total_current_class - target_image_count) / target_image_count * 100
            print(f"  -> Generated {len(fused_pairs_current_class)} images. Difference from target: {diff_percentage:.1f}%")

            if diff_percentage < ACCEPTABLE_DIFFERENCE_PERCENTAGE:
                final_pairs_for_this_class = fused_pairs_current_class
                break
            if total_current_class > target_image_count:
                max_th2 = th2
                th2 = (th2 + min_th2) / 2
            else:
                min_th2 = th2
                th2 = (th2 + max_th2) / 2
            if abs(max_th2 - min_th2) < 1.0:
                final_pairs_for_this_class = fused_pairs_current_class
                break

        class_output_dir = os.path.join(MAIN_OUTPUT_DIR, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        total_to_fuse = len(final_pairs_for_this_class)
        print_progress_bar(
            0, total_to_fuse, prefix='  Fusing:', length=40,
            event={"type": "fuse_progress", "cls": class_name, "phase": "fuse",
                   "class_idx": class_idx, "num_classes": num_classes}
        )
        # At start of fuse: compare contributes full 50% for this class
        overall = ((class_idx * 100.0) + 50.0 + 0.0) / float(num_classes)
        emit_event(type="overall_progress", percent=clamp(overall), cls=class_name,
                   class_idx=class_idx, num_classes=num_classes)

        generated_so_far = 0
        denom = max(1, total_to_fuse)
        for j, (p1, p2) in enumerate(final_pairs_for_this_class):
            outp = os.path.join(class_output_dir, f"fused_{class_name}_{j}.png")
            fuse_images(p1, p2, outp)
            generated_so_far += 1
            print_progress_bar(
                j + 1, total_to_fuse, prefix='  Fusing:', length=40,
                event={"type": "fuse_progress", "cls": class_name, "phase": "fuse",
                       "class_idx": class_idx, "num_classes": num_classes}
            )
            fuse_percent = 100.0 * ((j + 1) / float(denom))
            overall = ((class_idx * 100.0) + 50.0 + 0.5 * fuse_percent) / float(num_classes)
            emit_event(type="overall_progress", percent=clamp(overall), cls=class_name,
                       class_idx=class_idx, num_classes=num_classes)
            emit_event(type="generated", cls=class_name, generated_so_far=generated_so_far)

        class_results[class_name] = {
            'original': num_original_current,
            'generated': len(final_pairs_for_this_class)
        }
        emit_event(type="class_done", cls=class_name,
                   original=num_original_current,
                   generated=len(final_pairs_for_this_class),
                   total_after=num_original_current + len(final_pairs_for_this_class))

    overall_end_time = time.time()

    # ---------- Final summary ----------
    summary_filename = f"{dataset_name}_augmentation_summary.txt"
    summary_dir = "augmentation_results"
    os.makedirs(summary_dir, exist_ok=True)
    summary_filepath = os.path.join(summary_dir, summary_filename)

    lines = []
    lines.append(f"--- Augmentation Summary for {dataset_name} ---")
    lines.append("=" * 50)
    for name, result in class_results.items():
        original_count = result['original']
        generated_count = result['generated']
        total_after = original_count + generated_count
        increase_perc = (generated_count / original_count * 100) if original_count > 0 else 0
        lines.append(f"\nClass: {name}")
        lines.append(f"  - Images before augmentation: {original_count}")
        lines.append(f"  - Generated images: {generated_count}")
        lines.append(f"  - Increase in images: {increase_perc:.2f}%")
        lines.append(f"  - Total images after augmentation: {total_after}")

    lines.append("\n" + "=" * 50)
    lines.append("--- Overall Performance ---")
    lines.append(f"Total processing time: {overall_end_time - overall_start_time:.2f} seconds")
    lines.append(f"Peak memory usage: {memory_tracker.get_peak_memory():.2f} MB")

    with open(summary_filepath, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n\n--- Dynamic Augmentation Process Complete ---")
    print(f"--- Summary saved to: '{summary_filepath}' ---")

    # Pin overall to 100% at the end
    emit_event(type="overall_progress", percent=100.0)
    emit_event(
        type="done",
        elapsed_seconds=overall_end_time - overall_start_time,
        peak_mb=memory_tracker.get_peak_memory(),
        summary_path=summary_filepath
    )

# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dataset-dir", required=False, default="../../data/organised_data/Dataset_B/Training")
    parser.add_argument("--main-output-dir", required=False, default="../../data/augmented_data/MSE/Dataset_B/Training")
    parser.add_argument("--initial-th1", type=float, default=500.0)
    parser.add_argument("--initial-th2", type=float, default=825.0)
    parser.add_argument("--acceptable-difference-percentage", type=float, default=10.0)
    args = parser.parse_args()

    main(
        ROOT_DATASET_DIR=args.root_dataset_dir,
        MAIN_OUTPUT_DIR=args.main_output_dir,
        INITIAL_TH1=args.initial_th1,
        INITIAL_TH2=args.initial_th2,
        ACCEPTABLE_DIFFERENCE_PERCENTAGE=args.acceptable_difference_percentage,
    )
