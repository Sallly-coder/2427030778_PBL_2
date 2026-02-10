"""
train_eval.py
=============
Full training pipeline:
  1. Scan data directory for .wav files (RAVDESS naming convention)
  2. Preprocess audio
  3. Extract features
  4. Train/evaluate multiple classifiers
  5. Save best model & print metrics

Run:
    python src/train_eval.py --data_dir data/ravdess_subset \
                              --emotions happy sad neutral \
                              --model random_forest \
                              --test_size 0.3
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)

from src.audio_processing import AudioProcessor
from src.feature_extraction import FeatureExtractor
from src.model import EmotionClassifier


# ------------------------------------------------------------------
# RAVDESS emotion map (3rd token in filename = emotion code)
# ------------------------------------------------------------------
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


# ------------------------------------------------------------------
# Dataset builder
# ------------------------------------------------------------------

def build_dataset(data_dir: str,
                  emotions_to_use: list[str],
                  processor: AudioProcessor,
                  extractor: FeatureExtractor,
                  max_files_per_class: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk data_dir, find .wav files, filter by emotion, extract features.

    Parameters
    ----------
    data_dir           : str   Root folder (e.g. data/ravdess_subset/)
    emotions_to_use    : list  e.g. ['happy', 'sad', 'neutral']
    processor          : AudioProcessor
    extractor          : FeatureExtractor
    max_files_per_class: int or None (cap for quick testing)

    Returns
    -------
    X : np.ndarray  (n_samples, n_features)
    y : np.ndarray  (n_samples,) of string labels
    """
    X_list, y_list = [], []
    skipped = 0
    class_counts = {e: 0 for e in emotions_to_use}

    wav_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".wav"):
                wav_files.append(os.path.join(root, f))

    print(f"Found {len(wav_files)} .wav files in '{data_dir}'")

    for filepath in tqdm(wav_files, desc="Extracting features"):
        filename  = os.path.basename(filepath)
        parts     = filename.replace(".wav", "").split("-")

        if len(parts) < 7:
            skipped += 1
            continue

        emotion_code  = parts[2]
        emotion_label = RAVDESS_EMOTIONS.get(emotion_code)

        if emotion_label not in emotions_to_use:
            continue

        if (max_files_per_class is not None and
                class_counts[emotion_label] >= max_files_per_class):
            continue

        try:
            signal, sr = processor.full_preprocess(filepath)
            features   = extractor.extract_all(signal, sr)
            X_list.append(features)
            y_list.append(emotion_label)
            class_counts[emotion_label] += 1
        except Exception as e:
            print(f"\n  [WARN] Failed on {filename}: {e}")
            skipped += 1

    print(f"\nDataset summary:")
    for emo, count in class_counts.items():
        print(f"  {emo:12s} : {count} samples")
    print(f"  Skipped   : {skipped} files")

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


# ------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------

def print_metrics(y_true, y_pred, classes):
    """Print accuracy + classification report."""
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))


def plot_confusion_matrix(y_true, y_pred, classes,
                           title: str = "Confusion Matrix",
                           save_path: str = None):
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved → {save_path}")
    plt.show()


def compare_models(X_train, y_train, X_test, y_test, classes):
    """Train all models and compare accuracy."""
    model_types = ["logistic_regression", "random_forest", "mlp", "svm"]
    results = []

    print("\n" + "="*60)
    print("  MODEL COMPARISON")
    print("="*60)

    for mt in model_types:
        try:
            clf = EmotionClassifier(model_type=mt)
            t0  = time.time()
            clf.train(X_train, y_train)
            train_time = time.time() - t0

            preds = clf.predict(X_test)
            acc   = accuracy_score(y_test, preds)
            results.append({"model": mt, "accuracy": acc,
                             "train_time_s": round(train_time, 2)})
            print(f"  {mt:25s}  acc={acc*100:.1f}%  ({train_time:.1f}s)")
        except Exception as e:
            print(f"  {mt:25s}  FAILED: {e}")

    print("="*60)
    return results


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run_pipeline(data_dir: str,
                 emotions: list[str],
                 model_type: str = "random_forest",
                 test_size: float = 0.3,
                 save_model: bool = True,
                 max_files: int = None):
    """
    Full end-to-end training and evaluation pipeline.
    """
    print("\n" + "🎙️  SPEECH EMOTION RECOGNITION — Training Pipeline ".center(60, "="))

    # 1. Init processors
    processor = AudioProcessor(sample_rate=22050, duration=3.0)
    extractor = FeatureExtractor(n_mfcc=40)

    # 2. Build dataset
    X, y = build_dataset(
        data_dir=data_dir,
        emotions_to_use=emotions,
        processor=processor,
        extractor=extractor,
        max_files_per_class=max_files
    )

    if len(X) == 0:
        print("\n[ERROR] No samples found. Check data_dir and emotion names.")
        return

    print(f"\nFeature matrix shape : {X.shape}")

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )
    print(f"Train set : {len(X_train)} samples")
    print(f"Test  set : {len(X_test)} samples")

    # 4. Compare all models (optional quick overview)
    compare_models(X_train, y_train, X_test, y_test, emotions)

    # 5. Train chosen model
    print(f"\nTraining selected model: '{model_type}' ...")
    clf = EmotionClassifier(model_type=model_type)
    clf.train(X_train, y_train)

    # 6. Evaluate
    y_pred = clf.predict(X_test)
    print_metrics(y_test, y_pred, clf.classes)

    # 7. Confusion matrix
    os.makedirs("outputs", exist_ok=True)
    plot_confusion_matrix(
        y_test, y_pred, clf.classes,
        title=f"Confusion Matrix — {model_type}",
        save_path=f"outputs/confusion_matrix_{model_type}.png"
    )

    # 8. Cross-validation (on full data)
    print("\nRunning 5-fold cross-validation on full dataset...")
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from src.model import MODEL_REGISTRY
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    cv_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    MODEL_REGISTRY[model_type])
    ])
    cv_scores = cross_val_score(cv_pipeline, X, y_enc, cv=5, scoring="accuracy")
    print(f"  CV Accuracy : {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    # 9. Save model
    if save_model:
        os.makedirs("models", exist_ok=True)
        clf.save(f"models/ser_{model_type}.pkl")

    print("\n✅ Pipeline complete.")
    return clf


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Speech Emotion Recognition – Training Script"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/ravdess_subset",
        help="Path to directory containing Actor_XX subdirectories with .wav files"
    )
    parser.add_argument(
        "--emotions", nargs="+",
        default=["happy", "sad", "neutral"],
        help="Emotion labels to include (e.g. --emotions happy sad neutral angry)"
    )
    parser.add_argument(
        "--model", type=str, default="random_forest",
        choices=["logistic_regression", "random_forest", "mlp", "svm"],
        help="Classifier to train"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.3,
        help="Fraction of data for test set (default: 0.3)"
    )
    parser.add_argument(
        "--max_files", type=int, default=None,
        help="Max files per class (use for quick testing, e.g. 50)"
    )
    parser.add_argument(
        "--no_save", action="store_true",
        help="Don't save the trained model"
    )

    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        emotions=args.emotions,
        model_type=args.model,
        test_size=args.test_size,
        save_model=not args.no_save,
        max_files=args.max_files
    )
