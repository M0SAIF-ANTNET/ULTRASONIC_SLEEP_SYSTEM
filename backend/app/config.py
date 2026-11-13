"""Central configuration. Every limit and threshold lives here."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# PostgreSQL in production, e.g.:
#   DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/sleepdb
# Falls back to a local SQLite file so the demo runs with zero setup.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'sleep_demo.db'}")

# ---- Safety limits -------------------------------------------------------
# Demo mode: these match the website sliders. When the real hardware limits
# are known, change ONLY these numbers.
LIMITS = {
    "frequency_khz": (100, 800),
    "amplitude_percent": (10, 100),
    "timer_minutes": (1, 60),
}
FREQUENCY_STEP = 10
WAVEFORMS = ("sine", "square", "triangle")

# ---- ML ------------------------------------------------------------------
QUIZ_FEATURES = ["sleepiness", "stress", "energy", "difficulty", "last_night"]
NUMERIC_TARGETS = ["frequency_khz", "amplitude_percent", "timer_minutes"]
WAVEFORM_TARGET = "waveform"

N_SYNTHETIC = 800
REAL_SAMPLE_WEIGHT = 5.0        # a real observation counts 5x a simulated one
MIN_NEW_REAL_FOR_RETRAIN = 10   # retrain after this many new validated sessions

# ---- Outcome validation --------------------------------------------------
# A real session is used for training only if the astronaut was comfortable
# and the combined outcome score is high enough.
MIN_COMFORT = 3
MIN_OUTCOME_SCORE = 0.65

# ---- Personalization -----------------------------------------------------
# Weight given to an astronaut's own best past sessions: n / (n + K).
PERSONAL_BLEND_K = 5
PERSONAL_TOP_N = 5
