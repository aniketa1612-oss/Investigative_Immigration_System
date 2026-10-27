# Intelligent Investigative Interview Assistant

A B.Tech capstone decision-support application for neutral immigration interviews at airports, border-entry points, visa offices, and consulates. It helps an authorized officer capture transcripts, generate context-aware follow-up questions, and prepare transcript-grounded reports. It does **not** decide admissibility, visa eligibility, immigration status, or deception.

## Current implementation status

**Current working flow:** implemented. After explicit consent, the browser records short audio segments and the local FastAPI service transcribes them with faster-whisper. Temporary audio is deleted after transcription and transcript text is stored in SQLite. The camera preview and MediaPipe facial-landmark model also use local assets.

**LLM generation:** follow-up questions and reports default to the stronger local `qwen3:14b` model through Ollama with strict structured output. No active internet connection is required after the dependencies and models have been prepared.

## Features

- Consent gate before creating an interview or requesting devices
- Persistent indicator whenever camera or microphone capture is active
- Raw media storage opt-in, disabled by default
- International language selection, including major Indian and global languages
- Airport/border-entry and visa-office interview modes
- Immigration-focused questions covering travel purpose, itinerary, accommodation, funding, supporting documents, and application history
- Interview create/start/pause/resume/end lifecycle
- Offline faster-whisper transcription with persisted transcript segments
- Manual transcript correction when local transcription is unavailable or inaccurate
- LLM-generated, transcript-grounded neutral follow-up questions and structured reports
- Printable report with limitations and disclaimer
- SQLite local database with PostgreSQL-ready SQLAlchemy configuration
- Alembic migration scaffold
- Deterministic end-to-end backend test

## Stack

- Frontend: React, Vite, MediaRecorder, locally bundled MediaPipe assets
- Backend: FastAPI, SQLAlchemy, Alembic, Pydantic
- Database: SQLite by default; set `DATABASE_URL` for PostgreSQL

## Run locally

From the project root:

```powershell
Copy-Item .env.example .env
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd ..
cd frontend
npm install
cd ..
ollama pull qwen3:14b
.\scripts\prepare_offline_models.ps1
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend
npm run dev
```

Open `http://localhost:5173`. API documentation is available at `http://localhost:8000/docs`.

The `pip install`, `npm install`, `ollama pull`, and model-preparation script require internet access once. After they complete, normal application use is offline. Keep Ollama running locally. A local system text-to-speech voice is also required if the optional “Read question aloud” button must work offline.

## Test

```powershell
cd backend
python -m pytest -q

cd ..\frontend
npm run build
```

## Environment variables

- `DATABASE_URL`: SQLAlchemy database URL
- `DEMO_MODE`: keep `true` until real components have been executed and verified
- `PERSIST_RAW_MEDIA`: defaults to `false`; the current skeleton does not implement raw-media persistence
- `OFFLINE_MODE`: defaults to `true`
- `LLM_PROVIDER`: `ollama` for local generation or `openai` for the OpenAI API
- `LLM_BASE_URL`: use `http://localhost:11434/v1` for local Ollama
- `LLM_API_KEY`: required only for the OpenAI provider; never expose it to the frontend
- `LLM_MODEL`: local model used for questions and reports; defaults to `qwen3:14b`
- `WHISPER_MODEL_PATH`: local faster-whisper model directory
- `WHISPER_DEVICE`: `cpu` by default; use `cuda` only on a configured NVIDIA system
- `WHISPER_COMPUTE_TYPE`: `int8` by default for CPU use
- `CORS_ORIGINS`: comma-separated allowed browser origins
- `VITE_API_URL`: frontend API base URL

## Data model

Investigator, Interview, InterviewEvent, EmotionObservation, BehaviouralObservation, Question, Transcript, Recommendation, BehaviouralTimelineEvent, and InterviewReport are defined with foreign keys, timestamps, and indexes.

## API summary

- `GET /api/health`, `GET /api/config`
- `POST/GET /api/interviews`, `GET /api/interviews/{id}`
- `POST /api/interviews/{id}/lifecycle/start|pause|resume|end`
- `POST /api/interviews/{id}/transcripts`
- `POST /api/interviews/{id}/audio-segments`
- `POST /api/interviews/{id}/questions`
- `POST /api/interviews/{id}/report`
- `WS /ws/interviews/{id}`

## Multilingual scope

The interface supports major Indian languages plus widely spoken international languages. Whisper provides broad multilingual coverage, but this is not evidence of equal accuracy. Real deployment requires representative evaluation across each language, accent, dialect, code-switching pattern, recording quality, gender, age, and region.

## Ethical and legal limitations

Face and voice are biometric data. Explicit consent, purpose limitation, retention controls, access controls, audit trails, and jurisdiction-specific laws may apply, including GDPR-style requirements and BIPA-style statutes. This student project is not a certified forensic, medical, or law-enforcement system.

Behavioural signals are culturally and contextually variable. Facial affect, vocal stress, response latency, or linguistic patterns cannot prove deception. Automated outputs must be treated as uncertain observations and reviewed by trained personnel. Bias, accessibility, security, and multilingual performance require independent validation before any real-world pilot.

## Track B build order

1. Add a face detector and facial-affect adapter; verify real frame output.
2. Evaluate the implemented faster-whisper path across supported languages, accents, dialects, and code-switching.
3. Add acoustic features and a documented speech-emotion model or clearly labelled heuristic.
4. Compute rolling per-interview baselines and explainable multimodal fusion.
5. Add a provider-agnostic LLM adapter with schema validation, one retry, and neutral fallback.
6. Add authentication, authorization, PostgreSQL, encryption, retention jobs, and audit logging before any authorized pilot.
# Live response translation

In **Live transcript**, keep **Translate responses for the officer** enabled and choose the officer's reading language (English by default). Select the interviewee's spoken language during setup. Each saved audio segment or manually entered statement keeps its original text and receives a separately labelled machine translation. Matching languages need no translation.

Translation uses the existing configured LLM; with Ollama it runs locally and needs no extra API key or package. Keep Ollama and the configured model available. Translation runs after transcription of each eight-second recording segment, so latency also depends on transcription and model speed; it is not word-by-word simultaneous interpretation. Failed translations offer Retry and do not block recording. Translations are cached separately in interview events, without replacing the original transcript or report text. Verify names, dates, amounts, and ambiguous wording with the speaker or an interpreter.

