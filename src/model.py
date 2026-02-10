"""
model.py
=========
Defines emotion classifiers as a clean OOP wrapper around scikit-learn.

Supported models:
    - LogisticRegression   (fast baseline)
    - RandomForest         (stronger ensemble)
    - MLPClassifier        (shallow neural net)
    - SVM                  (often strong on MFCC features)

Usage:
    from src.model import EmotionClassifier

    clf = EmotionClassifier(model_type="random_forest")
    clf.train(X_train, y_train)
    preds = clf.predict(X_test)
    clf.save("models/rf_model.pkl")
"""

import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------------
# Available model configurations
# ------------------------------------------------------------------

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        multi_class="auto",
        random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    ),
    "mlp": MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    ),
    "svm": SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        random_state=42
    ),
}


class EmotionClassifier:
    """
    A wrapper around scikit-learn classifiers with a StandardScaler pipeline.

    Parameters
    ----------
    model_type : str
        One of: 'logistic_regression', 'random_forest', 'mlp', 'svm'
    """

    def __init__(self, model_type: str = "random_forest"):
        if model_type not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from: {list(MODEL_REGISTRY.keys())}"
            )
        self.model_type = model_type
        self.label_encoder = LabelEncoder()

        # Pipeline: scale first, then classify
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    MODEL_REGISTRY[model_type])
        ])

        self._is_trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the classifier on training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : array-like of string labels (e.g. ['happy', 'sad', ...])
        """
        y_encoded = self.label_encoder.fit_transform(y)
        self.pipeline.fit(X, y_encoded)
        self._is_trained = True
        print(f"[EmotionClassifier] Trained '{self.model_type}' on "
              f"{X.shape[0]} samples, {X.shape[1]} features.")
        print(f"[EmotionClassifier] Classes: {list(self.label_encoder.classes_)}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict emotion labels.

        Returns
        -------
        predictions : np.ndarray of string labels
        """
        self._check_trained()
        y_pred_enc = self.pipeline.predict(X)
        return self.label_encoder.inverse_transform(y_pred_enc)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability estimates for each class.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
        """
        self._check_trained()
        return self.pipeline.predict_proba(X)

    def predict_single(self, feature_vector: np.ndarray) -> tuple[str, dict]:
        """
        Predict a single audio file's emotion + confidence scores.

        Parameters
        ----------
        feature_vector : np.ndarray of shape (n_features,)

        Returns
        -------
        emotion   : str   (e.g. "happy")
        scores    : dict  {emotion_label: probability}
        """
        X = feature_vector.reshape(1, -1)
        emotion = self.predict(X)[0]
        proba   = self.predict_proba(X)[0]
        scores  = dict(zip(self.label_encoder.classes_, proba))
        return emotion, scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Serialize the entire classifier (pipeline + label encoder)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "pipeline":      self.pipeline,
                "label_encoder": self.label_encoder,
                "model_type":    self.model_type,
            }, f)
        print(f"[EmotionClassifier] Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "EmotionClassifier":
        """Load a previously saved classifier."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls.__new__(cls)
        instance.pipeline      = data["pipeline"]
        instance.label_encoder = data["label_encoder"]
        instance.model_type    = data["model_type"]
        instance._is_trained   = True
        print(f"[EmotionClassifier] Loaded '{instance.model_type}' from {path}")
        return instance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_trained(self):
        if not self._is_trained:
            raise RuntimeError("Call .train() before predicting.")

    @property
    def classes(self):
        """Return list of emotion class names."""
        self._check_trained()
        return list(self.label_encoder.classes_)

    def __repr__(self):
        status = "trained" if self._is_trained else "untrained"
        return f"EmotionClassifier(model_type='{self.model_type}', status={status})"
