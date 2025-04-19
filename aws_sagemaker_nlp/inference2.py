import os
import joblib
import numpy as np
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder

from transformers import DistilBertTokenizer, DistilBertModel

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs.last_hidden_state
        cls_output = hidden_states[:, 0]
        return self.classifier(cls_output)

        
def model_fn(model_dir, context=None):
    # Load label classes
    # classes = np.load(os.path.join(model_dir, "classes.npy"), allow_pickle=True)
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    classes = label_encoder.classes_

    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)

    # Initialize model and load state_dict
    model = DistilBertClassifier(num_labels=len(classes))
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu"))
    model.eval()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "classes": classes
    }


def input_fn(input_data, content_type):
    import json
    if content_type == "application/json":
        data = json.loads(input_data)
        return data["inputs"] if isinstance(data, dict) else data
    raise ValueError("Unsupported content type")


def predict_fn(input_data, model_data):
    # model_data is the return from `model_fn`
    # input_data is the return from `input_fn`
    tokenizer = model_data["tokenizer"]
    model = model_data["model"]

    # Tokenize input text
    encodings = tokenizer(input_data, truncation=True, padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(encodings["input_ids"], encodings["attention_mask"])
        proba = F.softmax(outputs, dim=1)

    # Build result
    result = []
    for row in proba:
        # Convert to list of (class, rounded probability) pairs
        class_proba = [(cls, round(prob.item(), 4)) for cls, prob in zip(model_data['classes'], row)]
        # Sort by probability descending
        sorted_class_proba = sorted(class_proba, key=lambda x: x[1], reverse=True)
        # Create OrderedDict
        result.append(OrderedDict(sorted_class_proba))
    top_labels = [next(iter(od)) for od in result]
    
    return {"label": top_labels, "proba": result}


def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        import json
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
