"""Business logic: recommend, evaluate feedback, personalize, retrain."""
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy.orm import Session

from . import ml_model
from .config import (MIN_COMFORT, MIN_NEW_REAL_FOR_RETRAIN, MIN_OUTCOME_SCORE,
                     PERSONAL_BLEND_K, PERSONAL_TOP_N, QUIZ_FEATURES)
from .database import SleepSession
from .safety import clamp_settings


def _now():
    return datetime.now(timezone.utc)


def quiz_dict(s: SleepSession) -> dict:
    return {f: getattr(s, f"q_{f}") for f in QUIZ_FEATURES}


def settings_of(s: SleepSession, prefix: str) -> dict | None:
    if getattr(s, f"{prefix}_frequency_khz") is None:
        return None
    key = "timer_minutes"
    return {
        "frequency_khz": getattr(s, f"{prefix}_frequency_khz"),
        "amplitude_percent": getattr(s, f"{prefix}_amplitude_percent"),
        "timer_minutes": getattr(s, f"{prefix}_{key}"),
        "waveform": getattr(s, f"{prefix}_waveform"),
    }


# ---------- Personalization ----------

def personalize(db: Session, astronaut_id: str, base: dict) -> tuple[dict, float]:
    """Blend the global model with this astronaut's own best validated sessions.

    weight = n / (n + K): 0 with no history, grows as real sessions accumulate.
    Simple on purpose: the personal target is the mean of their top-scoring
    sessions. It reflects what they rated well, not proof of what caused it.
    """
    rows = (db.query(SleepSession)
            .filter(SleepSession.astronaut_id == astronaut_id, SleepSession.validated.is_(True))
            .order_by(SleepSession.outcome_score.desc())
            .limit(PERSONAL_TOP_N).all())
    n = len(rows)
    if n == 0:
        return base, 0.0
    w = n / (n + PERSONAL_BLEND_K)
    used = [settings_of(r, "actual") or settings_of(r, "approved") for r in rows]

    def mean(key):
        return sum(u[key] for u in used) / n

    blended = {
        "frequency_khz": (1 - w) * base["frequency_khz"] + w * mean("frequency_khz"),
        "amplitude_percent": (1 - w) * base["amplitude_percent"] + w * mean("amplitude_percent"),
        "timer_minutes": (1 - w) * base["timer_minutes"] + w * mean("timer_minutes"),
        "waveform": (Counter(u["waveform"] for u in used).most_common(1)[0][0]
                     if w > 0.5 else base["waveform"]),
    }
    return blended, round(w, 2)


def recommend(db: Session, astronaut_id: str, quiz: dict) -> SleepSession:
    raw = ml_model.predict_raw(quiz)
    blended, weight = personalize(db, astronaut_id, raw)
    safe = clamp_settings(**blended)  # safety constraints applied last

    s = SleepSession(
        astronaut_id=astronaut_id,
        rec_frequency_khz=safe["frequency_khz"],
        rec_amplitude_percent=safe["amplitude_percent"],
        rec_timer_minutes=safe["timer_minutes"],
        rec_waveform=safe["waveform"],
        model_version=ml_model.get_meta()["version"],
        personalization_weight=weight,
        **{f"q_{k}": v for k, v in quiz.items()},
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


# ---------- Feedback evaluation ----------

def outcome_score(latency_min: int, quality: int, awakenings: int, refreshed: int) -> float:
    """0..1 combined sleep outcome. Weights are a judgement call, not a medical standard."""
    lat = 1 - min(latency_min, 60) / 60
    qual = (quality - 1) / 4
    awk = 1 - min(awakenings, 5) / 5
    ref = (refreshed - 1) / 4
    return round(0.25 * lat + 0.35 * qual + 0.15 * awk + 0.25 * ref, 3)


def save_feedback(db: Session, s: SleepSession, fb: dict) -> dict:
    s.fb_latency_min = fb["latency_min"]
    s.fb_quality = fb["quality"]
    s.fb_awakenings = fb["awakenings"]
    s.fb_refreshed = fb["refreshed"]
    s.fb_comfort = fb["comfort"]
    s.feedback_at = _now()

    s.outcome_score = outcome_score(fb["latency_min"], fb["quality"],
                                    fb["awakenings"], fb["refreshed"])
    s.validated = bool(fb["comfort"] >= MIN_COMFORT and s.outcome_score >= MIN_OUTCOME_SCORE)
    rec, act = settings_of(s, "rec"), settings_of(s, "actual")
    s.followed_recommendation = (rec == act)
    db.commit()

    retrained = retrain_if_needed(db)
    return {
        "outcome_score": s.outcome_score,
        "validated_for_training": s.validated,
        "followed_recommendation": s.followed_recommendation,
        "retrained": retrained is not None,
    }


# ---------- Retraining ----------

def real_training_frame(db: Session) -> pd.DataFrame:
    rows = db.query(SleepSession).filter(SleepSession.validated.is_(True)).all()
    out = []
    for r in rows:
        used = settings_of(r, "actual") or settings_of(r, "approved")
        out.append({**quiz_dict(r), **used})
    return pd.DataFrame(out)


def retrain_if_needed(db: Session, force: bool = False) -> dict | None:
    real = real_training_frame(db)
    new_since = len(real) - ml_model.get_meta().get("n_real", 0)
    if force or new_since >= MIN_NEW_REAL_FOR_RETRAIN:
        return ml_model.train_model(real)
    return None


def model_status(db: Session) -> dict:
    meta = ml_model.get_meta()
    n_valid = db.query(SleepSession).filter(SleepSession.validated.is_(True)).count()
    n_fb = db.query(SleepSession).filter(SleepSession.feedback_at.isnot(None)).count()
    return {
        **meta,
        "real_sessions_with_feedback": n_fb,
        "validated_real_sessions": n_valid,
        "new_validated_since_last_training": n_valid - meta.get("n_real", 0),
        "retrain_threshold": MIN_NEW_REAL_FOR_RETRAIN,
    }


def session_to_dict(s: SleepSession) -> dict:
    return {
        "id": s.id,
        "astronaut_id": s.astronaut_id,
        "session_date": s.session_date.isoformat() if s.session_date else None,
        "quiz": quiz_dict(s),
        "recommended": settings_of(s, "rec"),
        "approved": settings_of(s, "approved"),
        "actual": settings_of(s, "actual"),
        "model_version": s.model_version,
        "personalization_weight": s.personalization_weight,
        "feedback": None if s.feedback_at is None else {
            "latency_min": s.fb_latency_min, "quality": s.fb_quality,
            "awakenings": s.fb_awakenings, "refreshed": s.fb_refreshed,
            "comfort": s.fb_comfort,
        },
        "outcome_score": s.outcome_score,
        "validated": s.validated,
    }
