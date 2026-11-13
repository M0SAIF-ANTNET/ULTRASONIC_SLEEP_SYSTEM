"""Database connection and the single sleep_sessions table."""
from datetime import date, datetime, timezone

from sqlalchemy import (Boolean, Column, Date, DateTime, Float, Integer,
                        String, create_engine)
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()


def _now():
    return datetime.now(timezone.utc)


class SleepSession(Base):
    """One row = one night. Quiz -> recommendation -> approval -> actual use -> feedback."""
    __tablename__ = "sleep_sessions"

    id = Column(Integer, primary_key=True)
    astronaut_id = Column(String(64), index=True, nullable=False)
    session_date = Column(Date, default=date.today)
    created_at = Column(DateTime(timezone=True), default=_now)

    # Pre-sleep quiz (1-5 each)
    q_sleepiness = Column(Integer, nullable=False)
    q_stress = Column(Integer, nullable=False)
    q_energy = Column(Integer, nullable=False)
    q_difficulty = Column(Integer, nullable=False)
    q_last_night = Column(Integer, nullable=False)

    # What the model recommended
    rec_frequency_khz = Column(Integer, nullable=False)
    rec_amplitude_percent = Column(Integer, nullable=False)
    rec_timer_minutes = Column(Integer, nullable=False)
    rec_waveform = Column(String(16), nullable=False)
    model_version = Column(String(32))
    personalization_weight = Column(Float, default=0.0)

    # What the astronaut approved
    approved_frequency_khz = Column(Integer)
    approved_amplitude_percent = Column(Integer)
    approved_timer_minutes = Column(Integer)
    approved_waveform = Column(String(16))
    approved_at = Column(DateTime(timezone=True))

    # What was actually running when START was pressed
    actual_frequency_khz = Column(Integer)
    actual_amplitude_percent = Column(Integer)
    actual_timer_minutes = Column(Integer)
    actual_waveform = Column(String(16))
    started_at = Column(DateTime(timezone=True))

    # Post-sleep feedback
    fb_latency_min = Column(Integer)
    fb_quality = Column(Integer)       # 1-5
    fb_awakenings = Column(Integer)
    fb_refreshed = Column(Integer)     # 1-5
    fb_comfort = Column(Integer)       # 1-5
    feedback_at = Column(DateTime(timezone=True))

    # Evaluation (computed when feedback arrives)
    outcome_score = Column(Float)      # 0..1
    validated = Column(Boolean, default=False)  # eligible for training
    followed_recommendation = Column(Boolean)   # actual == recommended?


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
