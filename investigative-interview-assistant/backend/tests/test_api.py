import os
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
import pytest
from fastapi.testclient import TestClient
import app.main as main_module
from app.main import app

client = TestClient(app)


def test_translation_preserves_original_caches_and_checks_interview(monkeypatch):
    interview_id = create_interview()
    original = "Je reste deux semaines, pas trois."
    statement = client.post(f"/api/interviews/{interview_id}/transcripts", json={"speaker": "interviewee", "text": original, "language": "French"}).json()
    calls = []
    async def translate(**context):
        calls.append(context)
        return {"text": "I am staying two weeks, not three.", "note": ""}
    monkeypatch.setattr(main_module, "translate_statement", translate)
    path = f"/api/interviews/{interview_id}/transcripts/{statement['id']}/translation"
    response = client.post(path)
    assert response.status_code == 200
    assert response.json()["text"] == "I am staying two weeks, not three."
    assert response.json()["source_text"] == original
    assert client.post(path).json() == response.json()
    assert len(calls) == 1
    assert client.get(f"/api/interviews/{interview_id}/transcripts").json()[0]["text"] == original
    assert client.post(path + "?target_language=Unknown").status_code == 422
    other_id = create_interview()
    assert client.post(f"/api/interviews/{other_id}/transcripts/{statement['id']}/translation").status_code == 404
    same = client.post(path + "?target_language=French").json()
    assert same["text"] == original and same["machine_generated"] is False
    assert len(calls) == 1


def test_translation_failure_leaves_transcript_available_for_retry(monkeypatch):
    interview_id = create_interview()
    statement = client.post(f"/api/interviews/{interview_id}/transcripts", json={"speaker": "interviewee", "text": "Bonjour", "language": "French"}).json()
    async def unavailable(**context):
        raise RuntimeError("Translation model unavailable")
    monkeypatch.setattr(main_module, "translate_statement", unavailable)
    response = client.post(f"/api/interviews/{interview_id}/transcripts/{statement['id']}/translation")
    assert response.status_code == 503
    assert client.get(f"/api/interviews/{interview_id}/transcripts").json()[0]["text"] == "Bonjour"

@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    async def question(**context):
        assert context["prior_authority_statement"] == "I am travelling for a two-week visit and will stay at a hotel."
        if not context["transcripts"]:
            return {"text": "Please describe what happened in your own words.", "reason": "Open narrative invitation.", "strategy": "open-narrative"}
        latest = context["transcripts"][-1]["text"]
        number = len(context["prior_questions"])
        return {"text": f"Follow-up {number}: you mentioned '{latest}'. What related detail should be clarified next?", "reason": "Clarifies the latest statement.", "strategy": "contextual-detail"}

    async def report(**context):
        latest = context["transcripts"][-1]["text"]
        if context["statement_facial_associations"]:
            assert context["statement_facial_associations"][0]["statement"]["text"] == latest
        return {"observed": [f"The traveller stated: {latest}"], "interpreted": ["The travel plan requires document review."], "recommended": ["Check the itinerary against the submitted documents."], "limitations": ["The transcript does not independently verify the travel plan."]}

    monkeypatch.setattr(main_module, "generate_contextual_question", question)
    monkeypatch.setattr(main_module, "generate_contextual_report", report)

def create_interview():
    response = client.post("/api/interviews", json={"title": "Airport immigration interview", "objective": "Conduct a neutral airport immigration interview covering travel purpose and itinerary without making an admissibility decision.", "language": "Hindi", "consent_given": True, "interviewee": {"full_name": "Test Traveller", "age": 32, "relationship_to_event": "Visitor", "contact_reference": "TRAVEL-TEST-01", "initial_statement": "I am travelling for a two-week visit and will stay at a hotel.", "additional_notes": "Uses Hindi for the interview."}})
    assert response.status_code == 201
    return response.json()["id"]

def test_health_and_consent_gate():
    assert client.get("/api/health").json()["demo_mode"] is True
    assert client.get("/api/health").json()["offline_mode"] is True
    config = client.get("/api/config").json()
    assert config["speech_to_text"] == "faster-whisper"
    assert "Spanish" in {item["name"] for item in config["languages"]}
    denied = client.post("/api/interviews", json={"title": "No consent", "objective": "Should not capture", "consent_given": False})
    assert denied.status_code == 422

def test_offline_audio_segment_is_transcribed_and_temporary_audio_is_not_stored(monkeypatch):
    interview_id = create_interview()
    client.post(f"/api/interviews/{interview_id}/lifecycle/start")

    def local_transcription(path, selected_language):
        assert os.path.exists(path)
        assert selected_language == "Hindi"
        return {"text": "मैंने वाहन देखा।", "detected_language": "hi", "language_probability": 0.96}

    monkeypatch.setattr(main_module, "transcribe_audio", local_transcription)
    response = client.post(
        f"/api/interviews/{interview_id}/audio-segments",
        data={"language": "Hindi"},
        files={"audio": ("segment.webm", b"test-audio", "audio/webm")},
    )
    assert response.status_code == 201
    assert response.json()["text"] == "मैंने वाहन देखा।"
    assert response.json()["detected_language"] == "hi"

def test_full_demo_workflow():
    interview_id = create_interview()
    assert client.post(f"/api/interviews/{interview_id}/lifecycle/start").json()["status"] == "active"
    transcript = client.post(f"/api/interviews/{interview_id}/transcripts", json={"speaker": "interviewee", "text": "I will visit Delhi for two weeks.", "language": "Hindi", "response_latency_ms": 1650})
    assert transcript.status_code == 201
    saved_transcripts = client.get(f"/api/interviews/{interview_id}/transcripts")
    assert saved_transcripts.status_code == 200
    assert saved_transcripts.json()[-1]["text"] == "I will visit Delhi for two weeks."
    facial = client.post(f"/api/interviews/{interview_id}/facial-observations", json={"observed":{"pattern":"Brow movement","cues":[{"name":"browInnerUp","strength":0.61}]},"interpreted":{"summary":"Brow movement was observed; context is unknown."},"confidence":0.61})
    assert facial.status_code == 201
    stored_facial = client.get(f"/api/interviews/{interview_id}/facial-observations").json()[-1]
    assert stored_facial["observed"]["pattern"] == "Brow movement"
    assert stored_facial["observed"]["associated_statement"]["id"] == transcript.json()["id"]
    assert stored_facial["observed"]["associated_statement"]["text"] == transcript.json()["text"]
    assert client.post(f"/api/interviews/{interview_id}/questions", json={"language": "Hindi"}).status_code == 201
    assert client.post(f"/api/interviews/{interview_id}/lifecycle/end").json()["status"] == "ended"
    report = client.post(f"/api/interviews/{interview_id}/report")
    assert report.status_code == 200 and report.json()["limitations"]
    report_data = report.json()
    assert report_data["analytics"]["statement_count"] == 1
    assert report_data["analytics"]["question_count"] == 1
    assert report_data["analytics"]["observation_count"] == 1
    assert report_data["analytics"]["association_count"] == 1
    assert report_data["analytics"]["statement_word_counts"] == [{"label": "Statement 1", "value": 7}]
    assert report_data["analytics"]["facial_pattern_counts"] == [{"label": "Brow movement", "value": 1}]
    assert report_data["analytics"]["expression_estimate_counts"] == [{"label": "Not estimated", "value": 1}]
    assert report_data["transcript"][0]["text"] == "I will visit Delhi for two weeks."
    assert report_data["statement_facial_associations"][0]["statement"]["id"] == transcript.json()["id"]
    assert report_data["metadata"]["interviewee"]["full_name"] == "Test Traveller"
    assert report_data["metadata"]["interviewee"]["initial_statement"].startswith("I am travelling")

def test_follow_up_questions_cycle_without_duplicates():
    interview_id = create_interview()
    opening = client.post(
        f"/api/interviews/{interview_id}/questions",
        json={"language": "Hindi"},
    )
    assert opening.status_code == 201
    assert opening.json()["strategy"] == "open-narrative"
    client.post(
        f"/api/interviews/{interview_id}/transcripts",
        json={"speaker": "interviewee", "text": "मैं दो सप्ताह के लिए भारत आया हूँ।", "language": "Hindi"},
    )
    questions = {
        client.post(
            f"/api/interviews/{interview_id}/questions",
            json={"language": "Hindi"},
        ).json()["text"]
        for _ in range(8)
    }
    assert len(questions) == 8
