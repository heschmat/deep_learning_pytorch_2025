import os
import json
from PIL import Image
from urllib.request import urlopen
from io import BytesIO

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


NUM_CLASSES = 133

# LABELS_PATH = os.path.join(os.path.dirname(__file__), "labels.json")
# Absolute path to the code directory
MODEL_DIR = os.environ.get('MODEL_DIR', '/opt/ml/model/')
logger.info(f"Listing files in model directory ({MODEL_DIR}):")
for root, dirs, files in os.walk(MODEL_DIR):
    for file in files:
        logger.info(f"XXX Found file: {os.path.join(root, file)}")


LABELS_PATH = os.path.join(MODEL_DIR, "code", "labels.json")
logger.info(f"Expected labels file path: {LABELS_PATH}")

# logger.info("Fetching the labels...")
# with open(LABELS_PATH, "r") as f:
#     LABELS = json.load(f)
# labels_sample = {k: v for k, v in list(LABELS.items())[:5]}
# logger.info(f'Labels fetched. Sample: {labels_sample}')

try:
    logger.info(f"Fetching the labels from: {LABELS_PATH}")
    with open(LABELS_PATH, "r") as f:
        LABELS = json.load(f)

    labels_sample = {k: v for k, v in list(LABELS.items())[:5]}
    logger.info(f'Labels fetched. Sample: {labels_sample}')

except Exception as e:
    logger.exception(f"🚨 Failed loading labels.json: {e}")
    LABELS = {}  # Prevent crash so you can get logs at least


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        # w = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        # self.model = models.efficientnet_b0(weights=w)
        self.model = models.efficientnet_b0(weights=None)

        # Freeze feature extractor
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Replace classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# def model_fn(model_dir):
#     model = EfficientNetModel(pretrained=False)
#     model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), map_location='cpu'))
#     model.eval()
#     return model


def model_fn(model_dir):
    logger.info(f"[model] Loading model from directory: {model_dir}")

    try:
        model_path = os.path.join(model_dir, "model.pth")
        logger.info(f"[model] Expected model file at: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[model] Model file not found at: {model_path}")

        model = EfficientNetModel()
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()

        logger.info("[model] Model loaded and ready for inference.")
        return model

    except Exception as e:
        logger.exception(f"[model] Failed to load model: {e}")
        raise RuntimeError(f"[model] Model loading failed: {e}")


# def input_fn(request_body, content_type='text/plain'):
#     image_path = request_body.strip()
#     image = Image.open(image_path).convert("RGB")
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])
#     return transform(image).unsqueeze(0)  # add batch dimension

def input_fn(request_body, content_type='text/plain'):
    path_or_url = request_body.strip()

    try:
        logger.info("Decoding payload ...")
        if isinstance(request_body, bytes):
            request_body = request_body.decode('utf-8')

        data = json.loads(request_body)
        path_or_url = data.get("image_url", "").strip()

        # If it's a URL, fetch it
        if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
            response = urlopen(path_or_url)
            image = Image.open(BytesIO(response.read())).convert("RGB")
        else:
            # Otherwise, treat it as local path
            image = Image.open(path_or_url).convert("RGB")

        logger.info("transforming the image data ...")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        return transform(image).unsqueeze(0)

    except Exception as e:
        raise ValueError(f"Could not load image from {path_or_url}: {e}")


def predict_fn(input_tensor, model):
    output = model(input_tensor)
    _, predicted = torch.max(output.data, 1)
    class_idx = predicted.item()
    return {
        "label": class_idx,
        "class_name": LABELS.get(str(class_idx), "Unknown")
    }


def output_fn(prediction, accept='application/json'):
    return json.dumps(prediction)
