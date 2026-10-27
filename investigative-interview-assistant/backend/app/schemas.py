from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class IntervieweeIntake(BaseModel):
    full_name: str = Field(min_length=2, max_length=160)
    age: int | None = Field(default=None, ge=1, le=130)
    relationship_to_event: str = Field(min_length=2, max_length=200)
    contact_reference: str | None = Field(default=None, max_length=200)
    initial_statement: str = Field(min_length=10, max_length=20000)
    additional_notes: str | None = Field(default=None, max_length=5000)


class InterviewCreate(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    objective: str = Field(min_length=5, max_length=2000)
    language: str = "English"
    consent_given: bool
    raw_media_opt_in: bool = False
    interviewee: IntervieweeIntake


class InterviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    objective: str
    language: str
    status: str
    consent_given: bool
    raw_media_opt_in: bool
    created_at: datetime


class TranscriptCreate(BaseModel):
    speaker: Literal["investigator", "interviewee"]
    text: str = Field(min_length=1, max_length=10000)
    language: str = "English"
    confidence: float = Field(default=1.0, ge=0, le=1)
    response_latency_ms: int | None = Field(default=None, ge=0)


class QuestionRequest(BaseModel):
    language: str = "English"


class FacialObservationCreate(BaseModel):
    observed: dict[str, Any]
    interpreted: dict[str, Any]
    confidence: float = Field(ge=0, le=1)
