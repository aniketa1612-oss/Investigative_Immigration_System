from functools import lru_cache
from pathlib import Path

from ..config import settings
from .languages import LANGUAGE_BY_NAME


@lru_cache(maxsize=1)
def _model():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Offline transcription is not installed. Run: python -m pip install -r requirements.txt") from exc

    model_path = Path(settings.whisper_model_path)
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parents[3] / model_path
    if not model_path.exists():
        raise RuntimeError(
            f"Offline Whisper model was not found at '{model_path}'. Run scripts\\prepare_offline_models.ps1 while online."
        )
    return WhisperModel(
        str(model_path),
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        local_files_only=True,
    )


def transcribe_audio(audio_path: str, selected_language: str) -> dict:
    language = LANGUAGE_BY_NAME.get(selected_language)
    segments, info = _model().transcribe(
        audio_path,
        language=language["code"] if language else None,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
    return {
        "text": text,
        "detected_language": info.language,
        "language_probability": float(info.language_probability),
    }
