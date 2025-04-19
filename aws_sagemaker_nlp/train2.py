import os
import argparse
import numpy as np
import pandas as pd
import random
import logging
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import DistilBertTokenizer, DistilBertModel

# ------------------------
# Setup Logging
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------
# Set Random Seed
# ------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------
# Dataset Class
# ------------------------
class NewsDataset(Dataset):
    def __init__(self, titles, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(titles, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ------------------------
# Model Class
# ------------------------
# class DistilBertClassifier(nn.Module):
#     def __init__(self, num_labels):
#         super().__init__()
#         self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(self.bert.config.hidden_size, num_labels)
#         )

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token
#         return self.classifier(cls_output)

class DistilBertClassifier(nn.Module):
    def __init__(self, num_labels, hidden_dim=256, dropout_prob=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        return self.classifier(cls_output)


# ------------------------
# Training Function
# ------------------------
def train_model(model, train_loader, val_loader, device, epochs, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss, correct_train, total_train = 0, 0, 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = epoch_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss, correct_val, total_val = 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        logger.info(f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs


# ------------------------
# Plotting
# ------------------------
def plot_metrics(train_losses, val_losses, train_accs, val_accs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Epochs start at 1
    epochs = list(range(1, len(train_losses) + 1))

    # Save metrics as CSV
    metrics_df = pd.DataFrame({
        "epoch": epochs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs
    })
    metrics_df.to_csv(os.path.join(output_dir, "training_metrics.csv"), index=False)

    # Plot with seaborn style
    # plt.style.use("seaborn-darkgrid")
    plt.style.use("ggplot")
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss Plot
    loss_ax.plot(epochs, train_losses, label='Train Loss', marker='o')
    loss_ax.plot(epochs, val_losses, label='Val Loss', marker='o')
    loss_ax.set_title("Loss over Epochs")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.legend()

    # Accuracy Plot
    acc_ax.plot(epochs, train_accs, label='Train Accuracy', marker='o')
    acc_ax.plot(epochs, val_accs, label='Val Accuracy', marker='o')
    acc_ax.set_title("Accuracy over Epochs")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.legend()

    loss_ax.set_xticks(epochs)
    acc_ax.set_xticks(epochs)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics.png"))
    logger.info(f"Plot saved: {os.path.exists(os.path.join(output_dir, 'training_metrics.png'))}")
    plt.close()


# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--train_file", type=str, default="/opt/ml/input/data/train/train.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--frac", type=float, default=1)
    args = parser.parse_args()

    try:
        set_seed(42)

        logger.info("Loading data...")
        df = pd.read_csv(args.train_file, low_memory=False)
        df = df.sample(frac=args.frac, random_state=42)

        label_encoder = LabelEncoder()
        df["label"] = label_encoder.fit_transform(df["CATEGORY"])

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df["TITLE"].tolist(), df["label"].tolist(), test_size=0.1, random_state=42
        )

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
        val_dataset = NewsDataset(val_texts, val_labels, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        model = DistilBertClassifier(num_labels=len(label_encoder.classes_))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Starting training...")
        train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, device, args.epochs, args.lr
        )

        logger.info("Saving artifacts...")
        os.makedirs(args.model_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pt"))
        tokenizer.save_pretrained(args.model_dir)
        joblib.dump(label_encoder, os.path.join(args.model_dir, "label_encoder.joblib"))

        plot_metrics(train_losses, val_losses, train_accs, val_accs, args.model_dir)
        logger.info("Training complete.")

    except Exception as e:
        logger.exception(f"Training failed: {e}")


if __name__ == "__main__":
    main()
