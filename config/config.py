from functools import lru_cache
from pathlib import Path
from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "Scribos"
    api_prefix:   str = "/api/v1"
    debug:        bool = False

    MODEL_NAME:      str
    LOCAL_MODEL_DIR: Path
    WORDS_FILE:      Path
    INDEX_FILE:      Path
    DRIVE_MODEL_URL: str

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",  # Esto es ignorado en Railway, pero útil en local
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
