from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Intelligent Investigative Interview Assistant"
    database_url: str = "sqlite:///./interviews.db"
    demo_mode: bool = True
    persist_raw_media: bool = False
    offline_mode: bool = True
    llm_provider: str = "ollama"
    llm_api_key: str | None = None
    llm_base_url: str | None = "http://localhost:11434/v1"
    llm_model: str = "qwen3:14b"
    whisper_model_path: str = "models/whisper/small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    max_audio_segment_mb: int = 12
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
