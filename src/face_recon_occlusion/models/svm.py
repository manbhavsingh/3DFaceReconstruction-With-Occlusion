"""
StandardScaler + RBF SVC classifier pipeline wrapper for geometry PAD.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_svm_pipeline(
    C=1.0,
    kernel="rbf",
    gamma="scale",
    class_weight="balanced",
    random_state=42
):
    """
    Build a scikit-learn Pipeline with StandardScaler and SVC.

    Args:
        C (float): Regularization parameter.
        kernel (str): Kernel type ('rbf', 'linear', etc.).
        gamma (str | float): Kernel coefficient.
        class_weight (str | dict): Class weighting strategy.
        random_state (int): Seed.

    Returns:
        Pipeline: Fitted/unfitted scikit-learn pipeline object.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                class_weight=class_weight,
                probability=True,
                random_state=random_state,
            ),
        ),
    ])


def train_svm(
    X_train,
    y_train,
    C=1.0,
    kernel="rbf",
    gamma="scale",
    class_weight="balanced",
    random_state=42
):
    """
    Fit SVM pipeline on training data strictly.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels (0=Bona Fide, 1=Spoof).
        C (float): Regularization parameter.
        kernel (str): Kernel type.
        gamma (str | float): Kernel coefficient.
        class_weight (str | dict): Class weight handling.
        random_state (int): Seed.

    Returns:
        Pipeline: Fitted pipeline.
    """
    pipeline = build_svm_pipeline(
        C=C,
        kernel=kernel,
        gamma=gamma,
        class_weight=class_weight,
        random_state=random_state,
    )
    pipeline.fit(X_train, y_train)
    return pipeline
