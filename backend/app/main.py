"""FastAPI app. The ML model only RECOMMENDS; it never touches the ESP32.
Run from backend/:  uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from . import ml_model, service
from .config import FRONTEND_DIR, LIMITS
from .database import Base, SleepSession, engine, get_db
from .safety import validate_settings
from .schemas import Feedback, RecommendRequest, Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    ml_model.get_meta()  # loads the model, or trains the initial (simulated) one
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
    return {"status": "ok", "limits": LIMITS}


@app.post("/api/recommend")
def recommend(req: RecommendRequest, db: Session = Depends(get_db)):
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
def approve(session_id: int, settings: Settings, db: Session = Depends(get_db)):
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
def start(session_id: int, settings: Settings, db: Session = Depends(get_db)):
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
def feedback(session_id: int, fb: Feedback, db: Session = Depends(get_db)):
    s = _get_session(db, session_id)
    if s.started_at is None:
        raise HTTPException(409, "Session was never started")
    if s.feedback_at is not None:
        raise HTTPException(409, "Feedback already submitted")
    return {"session_id": s.id, **service.save_feedback(db, s, fb.model_dump())}


@app.get("/api/astronauts/{astronaut_id}/pending")
def pending(astronaut_id: str, db: Session = Depends(get_db)):
    """Latest started session still waiting for morning feedback (or null)."""
    s = (db.query(SleepSession)
         .filter(SleepSession.astronaut_id == astronaut_id,
                 SleepSession.started_at.isnot(None),
                 SleepSession.feedback_at.is_(None))
         .order_by(SleepSession.id.desc()).first())
    return {"session": service.session_to_dict(s) if s else None}


@app.get("/api/astronauts/{astronaut_id}/sessions")
def history(astronaut_id: str, db: Session = Depends(get_db)):
    rows = (db.query(SleepSession).filter(SleepSession.astronaut_id == astronaut_id)
            .order_by(SleepSession.id.desc()).limit(30).all())
    return {"sessions": [service.session_to_dict(r) for r in rows]}


@app.get("/api/model/status")
def model_status(db: Session = Depends(get_db)):
    return service.model_status(db)


@app.post("/api/model/retrain")
def retrain(db: Session = Depends(get_db)):
    return service.retrain_if_needed(db, force=True)


# Serve the website from the same origin (must be mounted last).
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
