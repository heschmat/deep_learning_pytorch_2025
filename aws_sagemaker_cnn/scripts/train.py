import os
import sys
import copy
import argparse
import logging
import json
from typing import Tuple
from tqdm import tqdm

import boto3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms, datasets, models

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
)

# Constants
BUCKET_NAME = 'dog-breed-v2'
NUM_CLASSES = 133
DEFAULT_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 3

def get_labels_dict(bucket_name, prefix):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/')

    labels_dict = {}
    for page in pages:
        for obj in page.get('CommonPrefixes', []):
            folder_name = obj['Prefix'].split('/')[-2]
            number, name = folder_name.split('.', 1)
            labels_dict[int(number)] = name
    return labels_dict


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b0(weights=w)

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


def get_model(device: torch.device) -> nn.Module:
    model = EfficientNetModel()
    return model.to(device)


def test(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device) -> None:
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += inputs.size(0)

    logger.info(f"Testing Loss: {running_loss / total_samples:.4f}")
    logger.info(f"Testing Accuracy: {running_corrects / total_samples:.4f}")


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          criterion: nn.Module, optimizer: optim.Optimizer,
          device: torch.device, epochs: int = DEFAULT_EPOCHS) -> nn.Module:

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        for phase in ['train', 'valid']:
            model.train() if phase == 'train' else model.eval()

            loader = train_loader if phase == 'train' else val_loader
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in tqdm(loader, desc=f"{phase.title()}ing"):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples

            logger.info(f"{phase.title()} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                    logger.info("Early stopping triggered.")
                    model.load_state_dict(best_model_wts)
                    return model

    model.load_state_dict(best_model_wts)
    return model


def create_data_loaders(data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_data = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=test_transform)
    test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)

    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(test_data, batch_size=batch_size, shuffle=False),
        DataLoader(val_data, batch_size=batch_size, shuffle=False)
    )


def main(args):
    try:
        logger.info(f"Hyperparameters: LR={args.learning_rate}, Batch Size={args.batch_size}, Epochs={args.epochs}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader, test_loader, val_loader = create_data_loaders(args.data, args.batch_size)
        model = get_model(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.model.classifier.parameters(), lr=args.learning_rate)

        logger.info("Starting training...")
        model = train(model, train_loader, val_loader, criterion, optimizer, device, epochs=args.epochs)

        logger.info("Evaluating model...")
        test(model, test_loader, criterion, device)

        os.makedirs(args.model_dir, exist_ok=True)
        model_path = os.path.join(args.model_dir, "model.pth")
        torch.save(model.cpu().state_dict(), model_path)
        # Model saved to /opt/ml/model/model.pth.
        logger.info(f'Model saved to {os.path.join(args.model_dir, "model.pth")}.')

        # # ✅ Save labels
        # code_dir = os.path.join(args.model_dir, "code")
        # os.makedirs(code_dir, exist_ok=True)
        # labels_dict = get_labels_dict(bucket_name=BUCKET_NAME, prefix='train/')
        # labels_path = os.path.join(code_dir, "labels.json")
        # with open(labels_path, "w") as f:
        #     json.dump(labels_dict, f)
        # logger.info(f"Labels saved to {labels_path}.")

    except Exception as e:
        logger.exception(f"Training failed due to error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    main(parser.parse_args())
