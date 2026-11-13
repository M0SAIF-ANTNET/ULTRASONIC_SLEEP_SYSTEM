# Astronaut Sleep — ML Personalization

Adds a quiz → AI recommendation → approval → morning feedback loop to the existing
ultrasonic sleep website. The ESP32 WebSocket code is unchanged.

## Run

```bash
pip install -r requirements.txt
cd backend
uvicorn app.main:app --reload
# open http://localhost:8000
```

First start trains the initial model on **simulated** data (a few seconds).
Default database is a local SQLite file; for PostgreSQL set:

```bash
export DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/sleepdb
```

## Layout

```
backend/app/
  config.py      all limits and thresholds (edit hardware limits HERE only)
  safety.py      clamp model output / validate frontend input
  synthetic.py   SIMULATED data generator (rule-based, tagged source="synthetic")
  ml_model.py    Random Forest train / save / load / predict
  database.py    one table: sleep_sessions
  schemas.py     request validation
  service.py     recommend, personalize, evaluate feedback, retrain
  main.py        REST endpoints
frontend/index.html   original page + 3 new panels (quiz, recommendation, feedback)
```

## Workflow

1. Quiz (5 answers, 1–5) → `POST /api/recommend`
2. Model predicts frequency / amplitude / timer (RandomForestRegressor) and waveform
   (RandomForestClassifier). Blended with the astronaut's own best past sessions,
   then clamped by `safety.py`.
3. Astronaut approves → `POST /api/sessions/{id}/approve` → values loaded into the
   existing controls (which send them to the ESP32 as before).
4. START → `POST /api/sessions/{id}/start` records the settings actually running.
5. Morning feedback → `POST /api/sessions/{id}/feedback` computes an outcome score.
6. Sessions with comfort ≥ 3 and score ≥ 0.65 become "validated" and join training
   (weight 5× a simulated row). Retrain triggers after 10 new validated sessions.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | /api/recommend | quiz → recommendation + new session |
| POST | /api/sessions/{id}/approve | astronaut approval (safety-validated) |
| POST | /api/sessions/{id}/start | record actual settings used |
| POST | /api/sessions/{id}/feedback | save feedback, evaluate, maybe retrain |
| GET | /api/astronauts/{id}/pending | session waiting for feedback |
| GET | /api/astronauts/{id}/sessions | last 30 sessions |
| GET | /api/model/status | version, data counts, holdout metrics |
| POST | /api/model/retrain | force retrain |

## Database (`sleep_sessions`)

astronaut_id, session_date · quiz (5 ints) · recommended (4) + model_version +
personalization_weight · approved (4) + approved_at · actual (4) + started_at ·
feedback (5) + feedback_at · outcome_score, validated, followed_recommendation.

## Limitations (read before claiming results)

- The initial model learns an **invented rule**, not real physiology. Its holdout
  metrics measure how well it copies the simulation.
- Learning from validated sessions is **preference learning**: "settings that
  coincided with good nights". It does not prove the settings caused better sleep.
  The model never explores; if it keeps suggesting similar settings, it cannot
  discover better ones. Validating any benefit needs a controlled experiment
  (e.g. randomized or crossover nights, with a no-stimulation baseline).
- Safety limits currently equal the website slider ranges (demo mode). Set real
  hardware limits in `config.py` before using the real ESP32 path.
- No authentication: astronaut ID is a plain text field. Fine for a demo only.
- Outcome-score weights (`service.outcome_score`) are a judgement call.
