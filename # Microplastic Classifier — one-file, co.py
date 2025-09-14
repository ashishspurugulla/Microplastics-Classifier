# Microplastic Classifier — one-file, copy-paste, runs first try
# Classes: bead, fragment, film, fiber
# Features: size, color_value, roundness, elongation, texture
# Usage: python main.py
# - Auto-installs pandas & scikit-learn if missing
# - Auto-creates microplastics.csv if missing
# - Trains RandomForest, prints classification report & accuracy
# - Optional: interactive single prediction at the end

import os, sys, subprocess, importlib, math

def _ensure(pkg):
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(f"📦 Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg], stdout=sys.stdout)

for pkg in ("pandas", "scikit-learn", "numpy"):
    _ensure(pkg)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = "microplastics.csv"
FEATURES = ["size", "color_value", "roundness", "elongation", "texture"]
LABEL = "label"
CLASSES = ["bead", "fragment", "film", "fiber"]
RNG = np.random.default_rng(42)

def _clip(x, lo, hi):
    return float(np.clip(x, lo, hi))

def make_sample_csv(path=DATA_PATH, rows_per_class=30):
    """Create a realistic-ish fake dataset so the script runs first time."""
    rows = []

    # Beads: small, very round, low elongation, smoother
    for _ in range(rows_per_class):
        rows.append([
            _clip(RNG.normal(0.6, 0.2), 0.05, 2.0),     # size
            _clip(RNG.normal(140, 25), 30, 255),        # color_value
            _clip(RNG.normal(0.88, 0.06), 0.65, 1.0),   # roundness
            _clip(RNG.normal(0.12, 0.06), 0.00, 0.35),  # elongation
            _clip(RNG.normal(0.35, 0.10), 0.05, 0.7),   # texture
            "bead"
        ])

    # Fragments: irregular, mid roundness/elongation, rougher
    for _ in range(rows_per_class):
        rows.append([
            _clip(RNG.normal(1.8, 0.7), 0.1, 6.0),
            _clip(RNG.normal(110, 30), 20, 240),
            _clip(RNG.normal(0.45, 0.12), 0.15, 0.8),
            _clip(RNG.normal(0.45, 0.15), 0.1, 0.9),
            _clip(RNG.normal(0.7, 0.12), 0.2, 1.0),
            "fragment"
        ])

    # Films: thin sheets; moderate roundness, lower elongation, smoother
    for _ in range(rows_per_class):
        rows.append([
            _clip(RNG.normal(1.2, 0.6), 0.1, 5.0),
            _clip(RNG.normal(180, 35), 40, 255),
            _clip(RNG.normal(0.62, 0.12), 0.25, 0.95),
            _clip(RNG.normal(0.28, 0.12), 0.02, 0.7),
            _clip(RNG.normal(0.35, 0.12), 0.05, 0.8),
            "film"
        ])

    # Fibers: long/thin; low roundness, very high elongation
    for _ in range(rows_per_class):
        rows.append([
            _clip(RNG.normal(3.5, 1.5), 0.3, 12.0),
            _clip(RNG.normal(90, 25), 10, 220),
            _clip(RNG.normal(0.22, 0.08), 0.05, 0.5),
            _clip(RNG.normal(0.85, 0.08), 0.55, 0.99),
            _clip(RNG.normal(0.5, 0.15), 0.1, 0.95),
            "fiber"
        ])

    df = pd.DataFrame(rows, columns=FEATURES + [LABEL]).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"✅ Created sample dataset at {path} ({len(df)} rows)")

def load_dataset(path=DATA_PATH):
    if not os.path.exists(path):
        print("⚠️  microplastics.csv not found — generating a sample so you can run immediately...")
        make_sample_csv(path)
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}\nExpected: {FEATURES + [LABEL]}")
    return df

def train_and_report(df):
    X = df[FEATURES].copy()
    y = df[LABEL].astype(str).copy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.22, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print("\n📊 Classification Report")
    print(classification_report(y_te, y_pred, digits=3))
    print(f"✅ Accuracy: {acc:.3f}")
    return clf

def ask_float(prompt_text, lo=None, hi=None):
    while True:
        s = input(prompt_text).strip()
        if s == "":
            return None
        try:
            v = float(s)
            if lo is not None and v < lo:
                print(f"Too low (min {lo}). Try again.")
                continue
            if hi is not None and v > hi:
                print(f"Too high (max {hi}). Try again.")
                continue
            return v
        except ValueError:
            print("Enter a number (or press Enter to skip).")

def predict_one_interactive(clf):
    print("\n(Type Enter on an empty line to skip interactive prediction.)")
    go = input("Press Enter to do ONE quick prediction, or type anything to skip: ").strip()
    if go != "":
        return
    print("Enter measured features:")
    size = ask_float(" size: ")
    if size is None: 
        return
    color = ask_float(" color_value (0–255): ", 0, 255);       
    rnd   = ask_float(" roundness (0–1): ", 0, 1)
    elon  = ask_float(" elongation (0–1): ", 0, 1)
    txt   = ask_float(" texture (0–1): ", 0, 1)

    row = pd.DataFrame([[size, color, rnd, elon, txt]], columns=FEATURES)
    pred = clf.predict(row)[0]
    # Show simple class probabilities if available
    try:
        proba = clf.predict_proba(row)[0]
        probs = {cls: round(float(p), 3) for cls, p in zip(clf.classes_, proba)}
    except Exception:
        probs = None
    print(f"\n🔮 Predicted class: {pred}")
    if probs:
        print(f"   Probabilities: {probs}")

def main():
    print("🔧 Microplastic Classifier (one-file)")
    df = load_dataset(DATA_PATH)
    clf = train_and_report(df)
    predict_one_interactive(clf)
    print("\nDone. (Use your own microplastics.csv later with the same columns to retrain.)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ Error:", e)
        sys.exit(1)
