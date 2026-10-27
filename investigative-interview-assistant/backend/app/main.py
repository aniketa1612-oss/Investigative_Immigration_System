from datetime import datetime, timezone
from collections import Counter
import os
import tempfile
import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from openai import AuthenticationError, NotFoundError as OpenAINotFoundError, RateLimitError
from sqlalchemy import select
from sqlalchemy.orm import Session
from .config import settings
from .database import Base, engine, get_db
from .models import BehaviouralObservation, Interview, IntervieweeProfile, InterviewEvent, InterviewReport, Question, Transcript
from .schemas import FacialObservationCreate, InterviewCreate, InterviewOut, QuestionRequest, TranscriptCreate
from .services.languages import LANGUAGE_BY_NAME, SUPPORTED_LANGUAGES
from .services.llm import generate_contextual_question, generate_contextual_report, translate_statement
from .services.transcription import transcribe_audio

Base.metadata.create_all(bind=engine)
app = FastAPI(title=settings.app_name, version="0.1.0", description="Student decision-support prototype; never a lie detector.")
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in settings.cors_origins.split(",")], allow_methods=["*"], allow_headers=["*"])


class SocketHub:
    def __init__(self): self.connections: dict[int, set[WebSocket]] = {}
    async def connect(self, key, ws):
        await ws.accept(); self.connections.setdefault(key, set()).add(ws)
    def disconnect(self, key, ws): self.connections.get(key, set()).discard(ws)
    async def publish(self, key, event, payload):
        dead = []
        for ws in self.connections.get(key, set()):
            try: await ws.send_json({"event": event, "interview_id": key, "payload": payload})
            except Exception: dead.append(ws)
        for ws in dead: self.disconnect(key, ws)


hub = SocketHub()


def require_interview(db: Session, interview_id: int):
    row = db.get(Interview, interview_id)
    if not row: raise HTTPException(404, "Interview not found")
    return row


def raise_llm_error(exc: Exception):
    if isinstance(exc, RuntimeError):
        raise HTTPException(503, str(exc)) from exc
    if settings.llm_provider == "ollama":
        if isinstance(exc, httpx.ConnectError):
            raise HTTPException(503, "Ollama is not reachable. Start Ollama and confirm it is listening on localhost:11434.") from exc
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            raise HTTPException(503, f"Ollama rejected the request (HTTP {status}). Confirm model '{settings.llm_model}' is installed.") from exc
        raise HTTPException(502, f"Ollama could not complete the request with model '{settings.llm_model}'.") from exc
    if isinstance(exc, AuthenticationError):
        raise HTTPException(503, "OpenAI rejected the API key. Create a valid key and update LLM_API_KEY.") from exc
    if isinstance(exc, RateLimitError):
        code = getattr(exc, "code", None)
        if code == "credit_balance_exhausted":
            raise HTTPException(503, "OpenAI API credits are exhausted. Add API credits in OpenAI billing, wait a few minutes, and try again.") from exc
        raise HTTPException(503, "OpenAI rate or usage limit reached. Check API billing and usage limits, then try again.") from exc
    if isinstance(exc, OpenAINotFoundError):
        raise HTTPException(503, f"OpenAI model '{settings.llm_model}' is unavailable to this API project.") from exc
    raise HTTPException(502, "OpenAI could not complete the request. Check the model, quota, and network connection.") from exc


@app.get("/api/health")
def health(): return {"status": "ok", "demo_mode": settings.demo_mode, "offline_mode": settings.offline_mode, "raw_media_persistence": settings.persist_raw_media}


@app.get("/api/config")
def config(): return {"demo_mode": settings.demo_mode, "offline_mode": settings.offline_mode, "languages": SUPPORTED_LANGUAGES, "raw_media_default": False, "llm_configured": settings.llm_provider == "ollama" or bool(settings.llm_api_key), "llm_provider": settings.llm_provider, "llm_model": settings.llm_model, "speech_to_text": "faster-whisper", "disclaimer": "Generated content must be checked against the transcript and does not prove deception."}


@app.post("/api/interviews", response_model=InterviewOut, status_code=201)
def create_interview(body: InterviewCreate, db: Session = Depends(get_db)):
    if not body.consent_given: raise HTTPException(422, "Explicit consent is required before capture begins")
    if body.language not in LANGUAGE_BY_NAME: raise HTTPException(422, "Unsupported language selection")
    interview_data = body.model_dump(exclude={"interviewee"})
    row = Interview(**interview_data); db.add(row); db.flush()
    db.add(IntervieweeProfile(interview_id=row.id, **body.interviewee.model_dump()))
    db.commit(); db.refresh(row); return row


@app.get("/api/interviews", response_model=list[InterviewOut])
def list_interviews(db: Session = Depends(get_db)): return list(db.scalars(select(Interview).order_by(Interview.created_at.desc())))


@app.get("/api/interviews/{interview_id}", response_model=InterviewOut)
def read_interview(interview_id: int, db: Session = Depends(get_db)): return require_interview(db, interview_id)


@app.post("/api/interviews/{interview_id}/lifecycle/{action}")
async def lifecycle(interview_id: int, action: str, db: Session = Depends(get_db)):
    transitions = {"start": ({"created"}, "active"), "pause": ({"active"}, "paused"), "resume": ({"paused"}, "active"), "end": ({"active", "paused"}, "ended")}
    if action not in transitions: raise HTTPException(404, "Unknown lifecycle action")
    row = require_interview(db, interview_id); valid, target = transitions[action]
    if row.status not in valid: raise HTTPException(409, f"Cannot {action} an interview with status '{row.status}'")
    row.status = target; now = datetime.now(timezone.utc)
    if action == "start": row.started_at = now
    if action == "end": row.ended_at = now
    db.add(InterviewEvent(interview_id=interview_id, event_type=f"interview_{target}", payload={"demo_mode": settings.demo_mode})); db.commit()
    await hub.publish(interview_id, "interview_status", {"status": target}); return {"id": interview_id, "status": target}


@app.post("/api/interviews/{interview_id}/transcripts", status_code=201)
async def add_transcript(interview_id: int, body: TranscriptCreate, db: Session = Depends(get_db)):
    require_interview(db, interview_id); row = Transcript(interview_id=interview_id, **body.model_dump()); db.add(row); db.commit(); db.refresh(row)
    payload = {"id": row.id, **body.model_dump(), "created_at": row.created_at.isoformat(), "demo_mode": settings.demo_mode}
    await hub.publish(interview_id, "transcript_update", payload); return payload


@app.post("/api/interviews/{interview_id}/audio-segments", status_code=201)
async def transcribe_audio_segment(
    interview_id: int,
    audio: UploadFile = File(...),
    language: str = Form("English"),
    db: Session = Depends(get_db),
):
    interview = require_interview(db, interview_id)
    if interview.status != "active":
        raise HTTPException(409, "Audio transcription requires an active interview")
    if language not in LANGUAGE_BY_NAME:
        raise HTTPException(422, "Unsupported language selection")
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(415, "Upload an audio recording")

    limit = settings.max_audio_segment_mb * 1024 * 1024
    data = await audio.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"Audio segment exceeds {settings.max_audio_segment_mb} MB")
    suffix = os.path.splitext(audio.filename or "segment.webm")[1] or ".webm"
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
            temporary.write(data)
            path = temporary.name
        try:
            result = await run_in_threadpool(transcribe_audio, path, language)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:
            # Return a readable API error instead of dropping the browser request
            # with an opaque "Failed to fetch" message.
            raise HTTPException(503, f"Offline transcription failed: {exc}") from exc
    finally:
        if path:
            try: os.unlink(path)
            except FileNotFoundError: pass

    if not result["text"]:
        return {"saved": False, **result}
    row = Transcript(
        interview_id=interview_id,
        speaker="interviewee",
        text=result["text"],
        language=language,
        confidence=result["language_probability"],
    )
    db.add(row); db.commit(); db.refresh(row)
    payload = {
        "id": row.id,
        "speaker": row.speaker,
        "text": row.text,
        "language": row.language,
        "confidence": row.confidence,
        "detected_language": result["detected_language"],
        "created_at": row.created_at.isoformat(),
        "saved": True,
        "demo_mode": settings.demo_mode,
    }
    await hub.publish(interview_id, "transcript_update", payload)
    return payload


@app.get("/api/interviews/{interview_id}/transcripts")
def list_transcripts(interview_id: int, db: Session = Depends(get_db)):
    require_interview(db, interview_id)
    rows = db.scalars(
        select(Transcript)
        .where(Transcript.interview_id == interview_id)
        .order_by(Transcript.created_at, Transcript.id)
    ).all()
    return [
        {
            "id": row.id,
            "speaker": row.speaker,
            "text": row.text,
            "language": row.language,
            "confidence": row.confidence,
            "response_latency_ms": row.response_latency_ms,
            "created_at": row.created_at,
        }
        for row in rows
    ]


@app.post("/api/interviews/{interview_id}/transcripts/{transcript_id}/translation")
async def translate_transcript(interview_id: int, transcript_id: int, target_language: str = "English", db: Session = Depends(get_db)):
    require_interview(db, interview_id)
    row = db.get(Transcript, transcript_id)
    if not row or row.interview_id != interview_id:
        raise HTTPException(404, "Statement not found in this interview")
    if target_language not in LANGUAGE_BY_NAME:
        raise HTTPException(422, "Unsupported translation language")
    # Cache translations separately from the original evidentiary transcript.
    events = db.scalars(select(InterviewEvent).where(InterviewEvent.interview_id == interview_id, InterviewEvent.event_type == "statement_translation")).all()
    for event in events:
        payload = event.payload
        if (payload.get("transcript_id") == row.id and payload.get("target_language") == target_language
                and payload.get("source_text") == row.text and payload.get("model") == settings.llm_model):
            return payload
    if row.language == target_language:
        translated = {"text": row.text, "note": ""}
    else:
        try:
            translated = await translate_statement(text=row.text, source_language=row.language, target_language=target_language)
        except Exception as exc:
            raise_llm_error(exc)
    payload = {"transcript_id": row.id, "source_text": row.text, "source_language": row.language,
               "target_language": target_language, **translated, "machine_generated": row.language != target_language,
               "model": settings.llm_model, "created_at": datetime.now(timezone.utc).isoformat()}
    db.add(InterviewEvent(interview_id=interview_id, event_type="statement_translation", payload=payload))
    db.commit()
    return payload


@app.post("/api/interviews/{interview_id}/facial-observations", status_code=201)
def add_facial_observation(interview_id: int, body: FacialObservationCreate, db: Session = Depends(get_db)):
    require_interview(db, interview_id)
    latest_statement = db.scalar(select(Transcript).where(Transcript.interview_id == interview_id, Transcript.speaker == "interviewee").order_by(Transcript.created_at.desc(), Transcript.id.desc()).limit(1))
    observed = dict(body.observed)
    observed["associated_statement"] = {"id": latest_statement.id, "text": latest_statement.text, "created_at": latest_statement.created_at.isoformat()} if latest_statement else None
    row = BehaviouralObservation(interview_id=interview_id, observed=observed, interpreted=body.interpreted, confidence=body.confidence)
    db.add(row); db.commit(); db.refresh(row)
    return {"id": row.id, "observed": row.observed, "interpreted": row.interpreted, "confidence": row.confidence, "created_at": row.created_at}


@app.get("/api/interviews/{interview_id}/facial-observations")
def list_facial_observations(interview_id: int, db: Session = Depends(get_db)):
    require_interview(db, interview_id)
    rows = db.scalars(select(BehaviouralObservation).where(BehaviouralObservation.interview_id == interview_id).order_by(BehaviouralObservation.created_at, BehaviouralObservation.id)).all()
    return [{"id": row.id, "observed": row.observed, "interpreted": row.interpreted, "confidence": row.confidence, "created_at": row.created_at} for row in rows]


@app.post("/api/interviews/{interview_id}/questions", status_code=201)
async def question(interview_id: int, body: QuestionRequest, db: Session = Depends(get_db)):
    interview = require_interview(db, interview_id)
    interviewee = db.scalar(select(IntervieweeProfile).where(IntervieweeProfile.interview_id == interview_id))
    transcript_rows = db.scalars(select(Transcript).where(Transcript.interview_id == interview_id).order_by(Transcript.created_at, Transcript.id)).all()
    prior_rows = db.scalars(select(Question).where(Question.interview_id == interview_id).order_by(Question.created_at, Question.id)).all()
    transcripts = [{"speaker": row.speaker, "text": row.text, "language": row.language} for row in transcript_rows]
    try:
        generated = await generate_contextual_question(language=body.language, objective=interview.objective, transcripts=transcripts, prior_questions=[row.text for row in prior_rows], prior_authority_statement=interviewee.initial_statement if interviewee else None)
    except Exception as exc:
        raise_llm_error(exc)
    result = {**generated, "confidence": 0.0, "language": body.language, "demo_mode": False, "model": settings.llm_model}
    row = Question(interview_id=interview_id, text=result["text"], reason=result["reason"], strategy=result["strategy"], language=result["language"], confidence=result["confidence"]); db.add(row); db.commit(); db.refresh(row)
    await hub.publish(interview_id, "question_generated", {"id": row.id, **result}); return {"id": row.id, **result}


@app.post("/api/interviews/{interview_id}/report")
async def report(interview_id: int, db: Session = Depends(get_db)):
    interview = require_interview(db, interview_id)
    interviewee = db.scalar(select(IntervieweeProfile).where(IntervieweeProfile.interview_id == interview_id))
    transcripts = db.scalars(select(Transcript).where(Transcript.interview_id == interview_id).order_by(Transcript.created_at, Transcript.id)).all()
    questions = db.scalars(select(Question).where(Question.interview_id == interview_id).order_by(Question.created_at, Question.id)).all()
    facial_observations = db.scalars(select(BehaviouralObservation).where(BehaviouralObservation.interview_id == interview_id).order_by(BehaviouralObservation.created_at, BehaviouralObservation.id)).all()
    transcript_data = [{"id": x.id, "speaker": x.speaker, "text": x.text, "language": x.language, "confidence": x.confidence, "response_latency_ms": x.response_latency_ms, "created_at": x.created_at.isoformat()} for x in transcripts]
    question_data = [{"id": x.id, "question": x.text, "reason": x.reason, "strategy": x.strategy, "language": x.language, "created_at": x.created_at.isoformat()} for x in questions]
    facial_data = [{"observed": row.observed, "interpreted": row.interpreted, "strength": row.confidence, "created_at": row.created_at.isoformat()} for row in facial_observations]
    statement_facial_associations = [
        {"statement": item["observed"]["associated_statement"], "facial_pattern": item["observed"].get("pattern"), "cues": item["observed"].get("cues", []), "strength": item["strength"], "expression_estimate": item["interpreted"].get("expression_estimate"), "expression_confidence": item["interpreted"].get("expression_confidence"), "observed_at": item["created_at"]}
        for item in facial_data if item["observed"].get("associated_statement")
    ]
    if not transcript_data:
        raise HTTPException(422, "A report requires at least one saved transcript statement")
    # Keep the model context focused. Full observations remain available in the
    # returned analytics, while the LLM receives only report-relevant fields.
    compact_facial_data = [
        {
            "pattern": item["observed"].get("pattern"),
            "cues": item["observed"].get("cues", []),
            "expression_estimate": item["interpreted"].get("expression_estimate"),
            "expression_confidence": item["interpreted"].get("expression_confidence"),
            "strength": item["strength"],
            "observed_at": item["created_at"],
        }
        for item in facial_data
    ]
    try:
        payload = await generate_contextual_report(title=interview.title, objective=interview.objective, language=interview.language, transcripts=transcript_data, questions=question_data, facial_observations=compact_facial_data, statement_facial_associations=statement_facial_associations, interviewee=({"full_name": interviewee.full_name, "age": interviewee.age, "relationship_to_event": interviewee.relationship_to_event, "contact_reference": interviewee.contact_reference, "initial_statement": interviewee.initial_statement, "additional_notes": interviewee.additional_notes} if interviewee else None))
    except Exception as exc:
        raise_llm_error(exc)
    row = db.scalar(select(InterviewReport).where(InterviewReport.interview_id == interview_id))
    if row:
        for key, value in payload.items(): setattr(row, key, value)
    else: row = InterviewReport(interview_id=interview_id, **payload); db.add(row)
    db.commit(); db.refresh(row)

    pattern_counts = Counter(item["observed"].get("pattern", "Unclassified movement") for item in facial_data)
    expression_counts = Counter(item["interpreted"].get("expression_estimate", "Not estimated") for item in facial_data)
    word_counts = [len(item["text"].split()) for item in transcript_data]
    duration_seconds = None
    if interview.started_at and interview.ended_at:
        duration_seconds = max(0, round((interview.ended_at - interview.started_at).total_seconds()))
    analytics = {
        "statement_count": len(transcript_data),
        "question_count": len(question_data),
        "observation_count": len(facial_data),
        "association_count": len(statement_facial_associations),
        "total_words": sum(word_counts),
        "average_statement_words": round(sum(word_counts) / len(word_counts), 1),
        "average_transcript_confidence": round(sum(item["confidence"] for item in transcript_data) / len(transcript_data), 3),
        "duration_seconds": duration_seconds,
        "facial_pattern_counts": [{"label": label, "value": count} for label, count in pattern_counts.most_common()],
        "expression_estimate_counts": [{"label": label, "value": count} for label, count in expression_counts.most_common()],
        "statement_word_counts": [{"label": f"Statement {index + 1}", "value": count} for index, count in enumerate(word_counts)],
        "facial_strength_timeline": [{"label": f"Observation {index + 1}", "value": round(item["strength"], 3), "pattern": item["observed"].get("pattern", "Unclassified movement"), "created_at": item["created_at"]} for index, item in enumerate(facial_data)],
    }
    return {
        "id": row.id,
        "interview": interview.title,
        "metadata": {"interview_id": interview.id, "objective": interview.objective, "language": interview.language, "status": interview.status, "consent_given": interview.consent_given, "raw_media_stored": interview.raw_media_opt_in and settings.persist_raw_media, "started_at": interview.started_at.isoformat() if interview.started_at else None, "ended_at": interview.ended_at.isoformat() if interview.ended_at else None, "generated_at": datetime.now(timezone.utc).isoformat(), "interviewee": ({"full_name": interviewee.full_name, "age": interviewee.age, "relationship_to_event": interviewee.relationship_to_event, "contact_reference": interviewee.contact_reference, "initial_statement": interviewee.initial_statement, "additional_notes": interviewee.additional_notes} if interviewee else None)},
        "analytics": analytics,
        "transcript": transcript_data,
        "questions": question_data,
        "statement_facial_associations": statement_facial_associations,
        **payload,
        "demo_mode": False,
        "model": settings.llm_model,
    }


@app.websocket("/ws/interviews/{interview_id}")
async def websocket(interview_id: int, ws: WebSocket):
    await hub.connect(interview_id, ws)
    try:
        await ws.send_json({"event": "connected", "interview_id": interview_id, "payload": {"demo_mode": settings.demo_mode}})
        while True:
            message = await ws.receive_json()
            if message.get("event") == "ping": await ws.send_json({"event": "pong", "interview_id": interview_id, "payload": {}})
    except (WebSocketDisconnect, ValueError): hub.disconnect(interview_id, ws)
