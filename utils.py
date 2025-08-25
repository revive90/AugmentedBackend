# utils.py
import cv2
import numpy as np

def fuse_images(image_path1: str, image_path2: str, output_path: str):
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