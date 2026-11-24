"""Reproducible Training & Cross-Validation Script for Mood Classifier."""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FeatureExtractor

MOOD_LABELS = ["High-Energy", "Chill / Lo-Fi", "Dark / Cinematic", "Upbeat"]

def generate_synthetic_audio(mood_idx: int, sr: int = 44100, duration: float = 2.0) -> np.ndarray:
    """Generates acoustic samples embodying mood acoustic signatures."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = np.zeros_like(t)

    if mood_idx == 0:  # High-Energy: heavy sub-kick, fast hats, high flux
        sig += 0.6 * np.sin(2 * np.pi * 55 * t) * (np.sin(2 * np.pi * 4 * t) > 0)
        sig += 0.3 * np.random.normal(0, 0.4, len(t)) * (np.sin(2 * np.pi * 8 * t) > 0.5)
        sig += 0.4 * np.sin(2 * np.pi * 880 * t)
    elif mood_idx == 1:  # Chill / Lo-Fi: soft warm chords, low pass, low flux
        sig += 0.4 * np.sin(2 * np.pi * 220 * t)
        sig += 0.3 * np.sin(2 * np.pi * 277.18 * t)
        sig += 0.2 * np.sin(2 * np.pi * 329.63 * t)
        sig += 0.05 * np.random.normal(0, 0.1, len(t))  # vinyl crackle
    elif mood_idx == 2:  # Dark / Cinematic: minor dissonance, slow sub-drone
        sig += 0.6 * np.sin(2 * np.pi * 40 * t)
        sig += 0.3 * np.sin(2 * np.pi * 110 * t)
        sig += 0.2 * np.sin(2 * np.pi * 116.54 * t)  # minor second dissonance
    elif mood_idx == 3:  # Upbeat: bright major chords, crisp transients
        sig += 0.4 * np.sin(2 * np.pi * 440 * t)
        sig += 0.3 * np.sin(2 * np.pi * 554.37 * t)
        sig += 0.3 * np.sin(2 * np.pi * 659.25 * t)
        sig += 0.2 * np.sin(2 * np.pi * 880 * t) * (np.sin(2 * np.pi * 2 * t) > 0)

    # Add gentle noise
    sig += 0.02 * np.random.normal(0, 0.05, len(t))
    return sig.astype(np.float32)

def train_mood_model(output_path: str = "models/mood_clf.joblib") -> Pipeline:
    fe = FeatureExtractor()
    X = []
    y = []

    print("[+] Synthesizing labeled acoustic dataset for 4 moods...")
    samples_per_class = 80
    for mood_idx in range(4):
        for _ in range(samples_per_class):
            audio = generate_synthetic_audio(mood_idx)
            feats = fe.extract(audio)
            X.append(feats)
            y.append(mood_idx)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)),
    ])

    print("[+] Training RandomForest mood classifier pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\n=== Model Evaluation Report ===")
    print(classification_report(y_test, y_pred, target_names=MOOD_LABELS))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"\n[✓] Serialized model saved to: {output_path}")
    return pipeline

if __name__ == "__main__":
    train_mood_model()
