"""SIMULATED training data.

These rows are invented from a hand-written rule, NOT measured from astronauts.
They only give the model a sensible starting point. Every row is tagged
source="synthetic" so it can never be confused with real observations.
"""
import numpy as np
import pandas as pd

from .config import N_SYNTHETIC, QUIZ_FEATURES
from .safety import clamp_settings


def generate_synthetic(n: int = N_SYNTHETIC, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    q = pd.DataFrame({f: rng.integers(1, 6, n) for f in QUIZ_FEATURES})

    # "Arousal": 0 = already very sleepy/relaxed, 1 = wired and hard to sleep.
    arousal = (
        (q.stress - 1) / 4
        + (q.energy - 1) / 4
        + (q.difficulty - 1) / 4
        + (5 - q.sleepiness) / 4
        + (5 - q.last_night) / 4
    ) / 5

    freq = 200 + 400 * arousal + rng.normal(0, 30, n)
    amp = 30 + 40 * arousal + rng.normal(0, 5, n)
    timer = 15 + 30 * arousal + rng.normal(0, 4, n)
    wave_score = arousal + rng.normal(0, 0.1, n)
    wave = np.where(wave_score < 0.4, "sine", np.where(wave_score < 0.65, "triangle", "square"))

    rows = [clamp_settings(f, a, t, w) for f, a, t, w in zip(freq, amp, timer, wave)]
    out = pd.concat([q, pd.DataFrame(rows)], axis=1)
    out["source"] = "synthetic"
    return out
