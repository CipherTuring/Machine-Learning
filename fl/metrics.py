"""Evaluación de un modelo: loss, accuracy, precision, recall y F1 (macro)."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []
    for x, y in loader:
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        y_true.append(y.numpy())
        y_pred.append(logits.argmax(dim=1).numpy())
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    # macro: promedio simple entre las 10 clases (cada dígito pesa igual, aunque tenga menos muestras)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return {
        "loss": total_loss / len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def format_metrics(m):
    return (f"loss={m['loss']:.4f}  acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
            f"rec={m['recall']:.4f}  f1={m['f1']:.4f}")
