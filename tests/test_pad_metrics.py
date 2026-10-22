"""
Tests for ISO/IEC 30107-3 PAD evaluation metrics and ROC-AUC calculation.
"""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from face_recon_occlusion.evaluation.pad_metrics import compute_pad_metrics
from face_recon_occlusion.models.svm import train_svm


def test_pad_metrics_analytical_cases():
    """Verify APCER, BPCER, ACER, accuracy, and confusion matrix against exact analytical cases."""
    # Case 1: Perfect classification
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    m = compute_pad_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["apcer"] == 0.0
    assert m["bpcer"] == 0.0
    assert m["acer"] == 0.0
    assert m["tn"] == 3 and m["tp"] == 3 and m["fp"] == 0 and m["fn"] == 0

    # Case 2: All Spoofs missed (FN = 3, TP = 0)
    y_pred_all_live = np.array([0, 0, 0, 0, 0, 0])
    m_missed = compute_pad_metrics(y_true, y_pred_all_live)
    assert m_missed["apcer"] == 1.0
    assert m_missed["bpcer"] == 0.0
    assert m_missed["acer"] == 0.5

    # Case 3: All Bona Fide rejected (FP = 3, TN = 0)
    y_pred_all_spoof = np.array([1, 1, 1, 1, 1, 1])
    m_rejected = compute_pad_metrics(y_true, y_pred_all_spoof)
    assert m_rejected["apcer"] == 0.0
    assert m_rejected["bpcer"] == 1.0
    assert m_rejected["acer"] == 0.5

    # Case 4: Known mix matching Primary Pilot test split (5 bona fide, 20 spoofs: TN=4, FP=1, FN=6, TP=14)
    y_true_mix = np.array([0]*5 + [1]*20)
    y_pred_mix = np.array([0]*4 + [1]*1 + [0]*6 + [1]*14)
    m_mix = compute_pad_metrics(y_true_mix, y_pred_mix)
    assert m_mix["tn"] == 4
    assert m_mix["fp"] == 1
    assert m_mix["fn"] == 6
    assert m_mix["tp"] == 14
    assert np.isclose(m_mix["bpcer"], 1 / 5)   # 0.2000
    assert np.isclose(m_mix["apcer"], 6 / 20)  # 0.3000
    assert np.isclose(m_mix["acer"], (0.20 + 0.30) / 2)  # 0.2500
    assert np.isclose(m_mix["accuracy"], 18 / 25)        # 0.7200


def test_roc_auc_continuous_decision_score_direction():
    """
    Verify that continuous decision scores accurately rank samples and that higher scores
    correspond to the positive class (1 = Spoof).
    """
    y_true = np.array([0, 0, 0, 1, 1, 1])
    # Scores strongly predicting class 1 (higher for 1, lower for 0)
    y_score_good = np.array([-2.5, -1.0, -0.5, 0.8, 1.5, 3.0])
    m = compute_pad_metrics(y_true, (y_score_good > 0).astype(int), y_score=y_score_good)
    assert m["roc_auc"] == 1.0

    # Scores backwards (higher for 0, lower for 1) should yield 0.0
    y_score_inverted = -y_score_good
    m_inv = compute_pad_metrics(y_true, (y_score_inverted > 0).astype(int), y_score=y_score_inverted)
    assert m_inv["roc_auc"] == 0.0


def test_backward_compatibility_y_prob():
    """Verify backward compatibility when caller provides y_prob instead of y_score."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_pad_metrics(y_true, y_pred, y_prob=y_prob)
    assert m["roc_auc"] == 1.0


def test_regression_svm_decision_score_vs_probability_auc():
    """
    Regression test for the issue where Platt scaling in SVC(probability=True)
    can produce inconsistent or inverted probabilities on small imbalanced datasets,
    whereas continuous decision_function preserves the correct positive-class direction.
    """
    rng = np.random.RandomState(42)
    # 60 spoofs (1), 15 bona fide (0) - mimicking the 75-sample training split
    X_train = rng.randn(75, 4)
    y_train = np.array([1]*60 + [0]*15)
    # Separate class 1 along the positive axis
    X_train[y_train == 1] += 0.8

    # Test set: 20 spoofs (1), 5 bona fide (0)
    X_test = rng.randn(25, 4)
    y_test = np.array([1]*20 + [0]*5)
    X_test[y_test == 1] += 0.8

    model = train_svm(X_train, y_train, random_state=42)

    # Verify decision scores align with positive class 1
    scores = model.decision_function(X_test)
    classes = list(model.named_steps["svm"].classes_)
    assert classes == [0, 1], f"Expected classes [0, 1], got {classes}"

    y_pred = model.predict(X_test)
    metrics_score = compute_pad_metrics(y_test, y_pred, y_score=scores)

    # Decision-function based AUC must reflect valid positive separation (> 0.5)
    assert metrics_score["roc_auc"] > 0.65, (
        f"Decision score ROC-AUC should indicate positive discrimination, got {metrics_score['roc_auc']}"
    )
