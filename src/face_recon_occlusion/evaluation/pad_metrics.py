"""
ISO/IEC 30107-3 Face Presentation Attack Detection (PAD) metrics computation module.

Label Convention:
  0 = Bona Fide (Live)
  1 = Presentation Attack (Spoof)

Metrics:
  - APCER (Attack Presentation Classification Error Rate) = FN / (FN + TP)
  - BPCER (Bona Fide Presentation Classification Error Rate) = FP / (TN + FP)
  - ACER (Average Classification Error Rate) = (APCER + BPCER) / 2
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - Confusion Matrix [[TN, FP], [FN, TP]]
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_pad_metrics(y_true, y_pred, y_prob=None):
    """
    Compute comprehensive PAD evaluation metrics adhering strictly to label conventions.

    Args:
        y_true (np.ndarray): True labels (0=Bona Fide, 1=Spoof).
        y_pred (np.ndarray): Predicted labels (0=Bona Fide, 1=Spoof).
        y_prob (np.ndarray, optional): Predicted probabilities for spoof class (1).

    Returns:
        dict: Summary of metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = float("nan")
    else:
        auc = float("nan")

    # Confusion matrix with labels=[0, 1]
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    # APCER: Spoofs misclassified as Live = FN / (FN + TP)
    apcer = float(FN / (FN + TP)) if (FN + TP) > 0 else float("nan")

    # BPCER: Live misclassified as Spoof = FP / (TN + FP)
    bpcer = float(FP / (TN + FP)) if (TN + FP) > 0 else float("nan")

    # ACER: Average of APCER and BPCER
    if not np.isnan(apcer) and not np.isnan(bpcer):
        acer = float((apcer + bpcer) / 2.0)
    else:
        acer = float("nan")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        "tn": int(TN),
        "fp": int(FP),
        "fn": int(FN),
        "tp": int(TP),
        "confusion_matrix": cm.tolist(),
    }
