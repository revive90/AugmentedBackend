# run_mse_baseline.py
import os
import shutil
import cv2
import numpy as np
import time
import sys
from datetime import datetime
import psutil

# --- 1. CONFIGURATION ---
# --- EDIT THIS SECTION FOR YOUR EXPERIMENT ---

# --- A. Dataset Path ---
dataset_name  = "Dataset_B"
ROOT_DATASET_DIR = "../../data/organised_data/Dataset_B/Training"

# --- B. Output Paths ---
MAIN_OUTPUT_DIR = "../../data/augmented_data/MSE/Dataset_B/Training"
RESULTS_OUTPUT_DIR = "augmentation_results"

# --- C. Initial Thresholds (from the OCMRI paper for Dataset A) ---
INITIAL_TH1 = 500.0
INITIAL_TH2 = 825.0
ACCEPTABLE_DIFFERENCE_PERCENTAGE = 10.0

# --- D. NEW: Class-specific MAX TH2 limits for quality control ---
# The dynamic adjustment will not be allowed to go above the 'max' value for each class.
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


# ----------------------------------------------------

# --- 2. MEMORY TRACKER CLASS ---
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


# --- 3. CORE HELPER FUNCTIONS ---
def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    if total == 0: total = 1
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: sys.stdout.write('\n')


def calculate_mse(image_path1, image_path2):
    try:
        img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None: return float('inf')
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
        err /= float(img1.shape[0] * img1.shape[1])
        return err
    except Exception:
        return float('inf')


def fuse_images(image_path1, image_path2, output_path):
    img1, img2 = cv2.imread(image_path1), cv2.imread(image_path2)
    if img1 is None or img2 is None: return
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    fused_image = np.zeros_like(img1)
    for i in range(img1.shape[1]):
        if i % 2 == 0:
            fused_image[:, i] = img1[:, i]
        else:
            fused_image[:, i] = img2[:, i]
    cv2.imwrite(output_path, fused_image)


def get_fused_pairs_for_mse(class_dir, thresholds, memory_tracker):
    image_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg'))]
    if len(image_paths) < 2: return []
    fused_pairs, total_comparisons, count = [], (len(image_paths) * (len(image_paths) - 1)) // 2, 0
    print_progress_bar(0, total_comparisons, prefix='  Comparing:', length=40)
    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            count += 1
            if count % 500 == 0 or count == total_comparisons:
                print_progress_bar(count, total_comparisons, prefix='  Comparing:', length=40)
                memory_tracker.track()
            path1, path2 = image_paths[i], image_paths[j]
            mse = calculate_mse(path1, path2)
            if thresholds['lower'] < mse < thresholds['upper']:
                fused_pairs.append((path1, path2))
    return fused_pairs


# --- 4. MAIN EXECUTION BLOCK ---
def main():
    overall_start_time = time.time()
    memory_tracker = MemoryTracker()

    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(ROOT_DATASET_DIR):
        print(f"Error: Dataset directory '{ROOT_DATASET_DIR}' not found.")
        return

    class_dirs = sorted([d.path for d in os.scandir(ROOT_DATASET_DIR) if d.is_dir()])
    print(f"Found {len(class_dirs)} classes: {[os.path.basename(d) for d in class_dirs]}")

    if os.path.exists(MAIN_OUTPUT_DIR): shutil.rmtree(MAIN_OUTPUT_DIR)
    os.makedirs(MAIN_OUTPUT_DIR)
    print(f"Created main output directory: '{MAIN_OUTPUT_DIR}'")

    class_results = {}

    # --- Process the First Class ---
    first_class_dir = class_dirs[0]
    class_name = os.path.basename(first_class_dir)
    print(f"\n--- Processing first class to set baseline: {class_name} ---")

    initial_thresholds = {'lower': INITIAL_TH1, 'upper': INITIAL_TH2}
    fused_pairs_first_class = get_fused_pairs_for_mse(first_class_dir, initial_thresholds, memory_tracker)

    class_output_dir = os.path.join(MAIN_OUTPUT_DIR, class_name)
    os.makedirs(class_output_dir)
    total_to_fuse = len(fused_pairs_first_class)
    print_progress_bar(0, total_to_fuse, prefix='  Fusing:', length=40)
    for i, (p1, p2) in enumerate(fused_pairs_first_class):
        fuse_images(p1, p2, os.path.join(class_output_dir, f"fused_{class_name}_{i}.png"))
        print_progress_bar(i + 1, total_to_fuse, prefix='  Fusing:', length=40)

    num_original_first = len(os.listdir(first_class_dir))
    num_fused_first = len(fused_pairs_first_class)
    target_image_count = num_original_first + num_fused_first

    class_results[class_name] = {'original': num_original_first, 'generated': num_fused_first}

    print(f"Generated {num_fused_first} images for '{class_name}'.")
    print(f"Target image count for all other classes is now: {target_image_count}")

    # --- Process Subsequent Classes ---
    for i in range(1, len(class_dirs)):
        current_class_dir = class_dirs[i]
        class_name = os.path.basename(current_class_dir)
        print(f"\n--- Dynamically processing class: {class_name} ---")

        num_original_current = len(os.listdir(current_class_dir))

        th2 = INITIAL_TH2
        min_th2 = CLASS_LIMITS.get(class_name, {}).get('min', INITIAL_TH1)
        max_th2 = CLASS_LIMITS.get(class_name, {}).get('max', INITIAL_TH2 * 2)
        print(f"  (Constraining Th2 search for this class between {min_th2} and {max_th2})")

        final_pairs_for_this_class = []
        while True:
            print(f"  Adjusting Th2 (current value: {th2:.2f})...")
            current_thresholds = {'lower': INITIAL_TH1, 'upper': th2}
            fused_pairs_current_class = get_fused_pairs_for_mse(current_class_dir, current_thresholds, memory_tracker)
            total_current_class = num_original_current + len(fused_pairs_current_class)

            diff_percentage = 100 if target_image_count == 0 else abs(
                total_current_class - target_image_count) / target_image_count * 100
            print(
                f"  -> Generated {len(fused_pairs_current_class)} images. Difference from target: {diff_percentage:.1f}%")

            if diff_percentage < ACCEPTABLE_DIFFERENCE_PERCENTAGE:
                final_pairs_for_this_class = fused_pairs_current_class
                break
            if total_current_class > target_image_count:
                max_th2 = th2;
                th2 = (th2 + min_th2) / 2
            else:
                min_th2 = th2;
                th2 = (th2 + max_th2) / 2
            if abs(max_th2 - min_th2) < 1.0:
                final_pairs_for_this_class = fused_pairs_current_class
                break

        class_output_dir = os.path.join(MAIN_OUTPUT_DIR, class_name)
        os.makedirs(class_output_dir)
        total_to_fuse = len(final_pairs_for_this_class)
        print_progress_bar(0, total_to_fuse, prefix='  Fusing:', length=40)
        for j, (p1, p2) in enumerate(final_pairs_for_this_class):
            fuse_images(p1, p2, os.path.join(class_output_dir, f"fused_{class_name}_{j}.png"))
            print_progress_bar(j + 1, total_to_fuse, prefix='  Fusing:', length=40)

        class_results[class_name] = {'original': num_original_current,
                                     'generated': len(final_pairs_for_this_class)}

    overall_end_time = time.time()

    # --- Create Final Summary Report ---
    summary_lines = []
    summary_filename = f"{dataset_name}_augmentation_summary.txt"
    summary_filepath = os.path.join(RESULTS_OUTPUT_DIR, summary_filename)

    summary_lines.append(f"--- Augmentation Summary for {dataset_name} ---")
    summary_lines.append("=" * 50)

    for name, result in class_results.items():
        original_count = result['original']
        generated_count = result['generated']
        total_after = original_count + generated_count
        increase_perc = (generated_count / original_count * 100) if original_count > 0 else 0

        summary_lines.append(f"\nClass: {name}")
        summary_lines.append(f"  - Images before augmentation: {original_count}")
        summary_lines.append(f"  - Generated images: {generated_count}")
        summary_lines.append(f"  - Increase in images: {increase_perc:.2f}%")
        summary_lines.append(f"  - Total images after augmentation: {total_after}")

    summary_lines.append("\n" + "=" * 50)
    summary_lines.append("--- Overall Performance ---")
    summary_lines.append(f"Total processing time: {overall_end_time - overall_start_time:.2f} seconds")
    summary_lines.append(f"Peak memory usage: {memory_tracker.get_peak_memory():.2f} MB")

    with open(summary_filepath, 'w') as f:
        f.write('\n'.join(summary_lines))

    print(f"\n\n--- Dynamic Augmentation Process Complete ---")
    print(f"--- Summary saved to: '{summary_filepath}' ---")


if __name__ == "__main__":
    main()