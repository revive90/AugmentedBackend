"""
Module summary:
- Provides a thin wrapper around a pretrained DINOv2 vision transformer to convert image files into fixed-length embeddings.

Public API:
- class FeatureExtractor:
  - __init__(model_name=..., device=...):
      Loads the image processor and model by name, moves the model to the specified device (GPU if available, otherwise CPU), and sets evaluation mode.
  - extract(image_path) -> np.ndarray | None:
      Opens the image (RGB), preprocesses it, runs a forward pass without gradients, mean-pools the last hidden states to a single feature vector, and returns it as a NumPy array on CPU. Returns None and logs a warning if processing fails.

Notes:
- DEVICE is chosen automatically based on torch.cuda availability.
- Embeddings are suitable for similarity, retrieval, or downstream ML tasks.
- Expected input: path to an image file; Output: 1D NumPy array of size equal to the model’s hidden dimension.
"""

import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import os
        
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/dinov2-base"

class FeatureExtractor:
        
    def __init__(self, model_name=MODEL_NAME, device=DEVICE):
        print("Loading DINOv2 model. This may take a moment...")
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        print(f"DINOv2 model loaded successfully on device: '{self.device}'.")

    def extract(self, image_path: str) -> np.ndarray | None:
        try:
            image = Image.open(image_path).convert("RGB")
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return features
        except Exception as e:
            print(f"Warning: Could not process {os.path.basename(image_path)}. Error: {e}")
            return None