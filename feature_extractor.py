# feature_extractor.py
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