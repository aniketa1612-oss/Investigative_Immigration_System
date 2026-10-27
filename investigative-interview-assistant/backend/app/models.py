from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base


def utcnow():
    return datetime.now(timezone.utc)


class Investigator(Base):
    __tablename__ = "investigators"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Interview(Base):
    __tablename__ = "interviews"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    objective: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(40), default="English")
    status: Mapped[str] = mapped_column(String(30), default="created", index=True)
    consent_given: Mapped[bool] = mapped_column(Boolean, default=False)
    raw_media_opt_in: Mapped[bool] = mapped_column(Boolean, default=False)
    investigator_id: Mapped[int | None] = mapped_column(ForeignKey("investigators.id"), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    events: Mapped[list["InterviewEvent"]] = relationship(cascade="all, delete-orphan")
    transcripts: Mapped[list["Transcript"]] = relationship(cascade="all, delete-orphan")
    observations: Mapped[list["BehaviouralObservation"]] = relationship(cascade="all, delete-orphan")
    interviewee: Mapped["IntervieweeProfile | None"] = relationship(back_populates="interview", cascade="all, delete-orphan", uselist=False)


class IntervieweeProfile(Base):
    __tablename__ = "interviewee_profiles"
    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id"), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(160))
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    relationship_to_event: Mapped[str] = mapped_column(String(200))
    contact_reference: Mapped[str | None] = mapped_column(String(200), nullable=True)
    initial_statement: Mapped[str] = mapped_column(Text)
    additional_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    interview: Mapped["Interview"] = relationship(back_populates="interviewee")


class InterviewEvent(Base):
    __tablename__ = "interview_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(60), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class EmotionObservation(Base):
    __tablename__ = "emotion_observations"
    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id"), index=True)
    modality: Mapped[str] = mapped_column(String(20))
    label: Mapped[str] = mapped_column(String(30))
    confidence: Mapped[float] = mapped_column(Float)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class BehaviouralObservation(Base):
    __tablename__ = "behavioural_observations"
    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id"), index=True)
    observed: Mapped[dict] = mapped_column(JSON)
    interpreted: Mapped[dict] = mapped_column(JSON)
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Question(Base):
    __tablename__ = "questions"
    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id"), index=True)
    text: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    strategy: Mapped[str] = mapped_column(String(80))
    language: Mapped[str] = mapped_column(String(40))
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Transcript(Base):
    __tablename__ = "transcripts"
    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id"), index=True)
    speaker: Mapped[str] = mapped_column(String(40))
    text: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(40))
    confidence: Mapped[float] = mapped_column(Float)
    response_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Recommendation(Base):
    __tablename__ = "recommendations"
    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id"), index=True)
    action: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    supporting_observations: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class BehaviouralTimelineEvent(Base):
    __tablename__ = "behavioural_timeline_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id"), index=True)
    category: Mapped[str] = mapped_column(String(50))
    observed: Mapped[str] = mapped_column(Text)
    interpreted: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class InterviewReport(Base):
    __tablename__ = "interview_reports"
    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id"), unique=True, index=True)
    observed: Mapped[list] = mapped_column(JSON, default=list)
    interpreted: Mapped[list] = mapped_column(JSON, default=list)
    recommended: Mapped[list] = mapped_column(JSON, default=list)
    limitations: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


Index("ix_emotion_interview_time", EmotionObservation.interview_id, EmotionObservation.observed_at)
