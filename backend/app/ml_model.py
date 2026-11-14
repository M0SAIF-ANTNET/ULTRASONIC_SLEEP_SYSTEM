"""Random Forest model: train, save, load, predict. No database code here."""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from .config import (BASE_DIR, MODEL_DIR, NUMERIC_TARGETS, QUIZ_FEATURES,
                     REAL_SAMPLE_WEIGHT, WAVEFORM_TARGET)
from .synthetic import generate_synthetic

MODEL_PATH = MODEL_DIR / "sleep_model.joblib"
# Initial model trained ONLY on simulated data, committed to the repo.
BUNDLED_PATH = BASE_DIR / "models" / "initial_model.joblib"
META_PATH = MODEL_DIR / "sleep_model_meta.json"
_bundle = None  # in-memory cache


def train_model(real_df: pd.DataFrame | None = None) -> dict:
    """Train on simulated data + validated real data. Returns metadata."""
    cols = QUIZ_FEATURES + NUMERIC_TARGETS + [WAVEFORM_TARGET, "source"]
    frames = [generate_synthetic()[cols]]
    n_real = 0
    if real_df is not None and len(real_df):
        frames.append(real_df.assign(source="real")[cols])
        n_real = len(real_df)
    data = pd.concat(frames, ignore_index=True)

    X = data[QUIZ_FEATURES]
    weights = np.where(data["source"] == "real", REAL_SAMPLE_WEIGHT, 1.0)

    def new_models():
        return (
            RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=0),
            RandomForestClassifier(n_estimators=100, min_samples_leaf=3, random_state=0),
        )

    # 1) Honest check on held-out rows
    tr, te = train_test_split(data.index, test_size=0.2, random_state=0)
    reg, clf = new_models()
    reg.fit(X.loc[tr], data.loc[tr, NUMERIC_TARGETS], sample_weight=weights[tr])
    clf.fit(X.loc[tr], data.loc[tr, WAVEFORM_TARGET], sample_weight=weights[tr])
    pred = reg.predict(X.loc[te])
    mae = {t: round(float(mean_absolute_error(data.loc[te, t], pred[:, i])), 2)
           for i, t in enumerate(NUMERIC_TARGETS)}
    acc = round(float(accuracy_score(data.loc[te, WAVEFORM_TARGET], clf.predict(X.loc[te]))), 3)

    # 2) Final model on all data
    reg, clf = new_models()
    reg.fit(X, data[NUMERIC_TARGETS], sample_weight=weights)
    clf.fit(X, data[WAVEFORM_TARGET], sample_weight=weights)

    meta = {
        "version": "rf-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_synthetic": int((data["source"] == "synthetic").sum()),
        "n_real": int(n_real),
        "holdout_mae": mae,
        "holdout_waveform_accuracy": acc,
        "note": ("Metrics are measured mostly against SIMULATED data. They show the model "
                 "learned the simulation, not that it improves real sleep."),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"reg": reg, "clf": clf, "meta": meta}, MODEL_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2))
    global _bundle
    _bundle = {"reg": reg, "clf": clf, "meta": meta}
    return meta


def _load() -> dict:
    """Load order: retrained model -> pre-trained simulated model shipped with the
    repo (instant, no training on cold start) -> train from scratch (slow last resort)."""
    global _bundle
    if _bundle is None:
        for path in (MODEL_PATH, BUNDLED_PATH):
            if path.exists():
                try:
                    _bundle = joblib.load(path)
                    return _bundle
                except Exception:
                    pass  # e.g. scikit-learn version mismatch -> try next option
        train_model()
    return _bundle


def get_meta() -> dict:
    return _load()["meta"]


def predict_raw(quiz: dict) -> dict:
    """Quiz answers -> raw (unsafe, unrounded) setting predictions."""
    b = _load()
    X = pd.DataFrame([{f: quiz[f] for f in QUIZ_FEATURES}])
    num = b["reg"].predict(X)[0]
    return {
        "frequency_khz": float(num[0]),
        "amplitude_percent": float(num[1]),
        "timer_minutes": float(num[2]),
        "waveform": str(b["clf"].predict(X)[0]),
    }
