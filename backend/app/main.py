"""FastAPI app. The ML model only RECOMMENDS; it never touches the ESP32.
Run from backend/:  uvicorn app.main:app --reload
"""
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.orm import Session

from . import ml_model, service
from .config import FRONTEND_DIR, LIMITS
from .database import Base, SessionLocal, SleepSession, engine, get_db
from .safety import validate_settings
from .schemas import Feedback, RecommendRequest, Settings


_ready = False


def ensure_ready():
    """Create tables and load the model. Runs once per process.

    Called lazily from the DB dependency (not only at startup) so it also works
    on serverless platforms. After a cold start /tmp is empty, so the model that
    ships with the repo (simulated data only) is loaded; it is retrained from
    the database once enough new validated sessions exist. Per-astronaut
    personalization reads the database directly, so it survives cold starts.
    """
    global _ready
    if _ready:
        return
    Base.metadata.create_all(engine)
    ml_model.get_meta()  # loads retrained or bundled model; no slow training here
    _ready = True


def db_session():
    ensure_ready()
    yield from get_db()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_ready()
    yield


app = FastAPI(title="Astronaut Sleep ML", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _get_session(db: Session, session_id: int) -> SleepSession:
    s = db.get(SleepSession, session_id)
    if s is None:
        raise HTTPException(404, "Session not found")
    return s


@app.get("/api/health")
def health():
    """Open this URL after deploying: it shows which part is broken."""
    info = {"status": "ok", "limits": LIMITS, "database": engine.dialect.name}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        info["database_ok"] = True
    except Exception as e:
        logging.exception("Database check failed")
        info.update(status="error", database_ok=False, database_error=type(e).__name__)
    try:
        info["model_version"] = ml_model.get_meta()["version"]
    except Exception as e:
        logging.exception("Model load failed")
        info.update(status="error", model_error=type(e).__name__)
    return info


@app.post("/api/recommend")
def recommend(req: RecommendRequest, db: Session = Depends(db_session)):
    s = service.recommend(db, req.astronaut_id, req.quiz.model_dump())
    return {
        "session_id": s.id,
        "recommendation": service.settings_of(s, "rec"),
        "source": "AI-generated (Random Forest)",
        "model_version": s.model_version,
        "personalization_weight": s.personalization_weight,
        "disclaimer": ("Initial model is trained on SIMULATED data. "
                       "Effect on real sleep has not been validated."),
    }


@app.post("/api/sessions/{session_id}/approve")
def approve(session_id: int, settings: Settings, db: Session = Depends(db_session)):
    s = _get_session(db, session_id)
    try:
        validate_settings(settings.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e))
    s.approved_frequency_khz = settings.frequency_khz
    s.approved_amplitude_percent = settings.amplitude_percent
    s.approved_timer_minutes = settings.timer_minutes
    s.approved_waveform = settings.waveform
    s.approved_at = service._now()
    db.commit()
    return {"session_id": s.id, "approved": service.settings_of(s, "approved")}


@app.post("/api/sessions/{session_id}/start")
def start(session_id: int, settings: Settings, db: Session = Depends(db_session)):
    """Frontend reports what was really running when START was pressed."""
    s = _get_session(db, session_id)
    if s.approved_at is None:
        raise HTTPException(409, "Settings must be approved before starting")
    try:
        validate_settings(settings.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e))
    s.actual_frequency_khz = settings.frequency_khz
    s.actual_amplitude_percent = settings.amplitude_percent
    s.actual_timer_minutes = settings.timer_minutes
    s.actual_waveform = settings.waveform
    s.started_at = service._now()
    db.commit()
    return {"session_id": s.id, "actual": service.settings_of(s, "actual")}


@app.post("/api/sessions/{session_id}/feedback")
def feedback(session_id: int, fb: Feedback, db: Session = Depends(db_session)):
    s = _get_session(db, session_id)
    if s.started_at is None:
        raise HTTPException(409, "Session was never started")
    if s.feedback_at is not None:
        raise HTTPException(409, "Feedback already submitted")
    return {"session_id": s.id, **service.save_feedback(db, s, fb.model_dump())}


@app.get("/api/astronauts/{astronaut_id}/pending")
def pending(astronaut_id: str, db: Session = Depends(db_session)):
    """Latest started session still waiting for morning feedback (or null)."""
    s = (db.query(SleepSession)
         .filter(SleepSession.astronaut_id == astronaut_id,
                 SleepSession.started_at.isnot(None),
                 SleepSession.feedback_at.is_(None))
         .order_by(SleepSession.id.desc()).first())
    return {"session": service.session_to_dict(s) if s else None}


@app.get("/api/astronauts/{astronaut_id}/sessions")
def history(astronaut_id: str, db: Session = Depends(db_session)):
    rows = (db.query(SleepSession).filter(SleepSession.astronaut_id == astronaut_id)
            .order_by(SleepSession.id.desc()).limit(30).all())
    return {"sessions": [service.session_to_dict(r) for r in rows]}


@app.get("/api/model/status")
def model_status(db: Session = Depends(db_session)):
    return service.model_status(db)


@app.post("/api/model/retrain")
def retrain(db: Session = Depends(db_session)):
    return service.retrain_if_needed(db, force=True)


# Serve the website from the same origin (must be mounted last).
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
