from functools import lru_cache
from pathlib import Path
from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    project_name: str = "Scribos"
    api_prefix:   str = "/api/v1"
    debug:        bool = False

    MODEL_NAME:      str = "dccuchile/bert-base-spanish-wwm-uncased"
    LOCAL_MODEL_DIR: Path = Path("/opt/models/beto_uncased")
    WORDS_FILE:      Path = Path("data/words.txt")
    INDEX_FILE:      Path = Path("data/index.json")

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",  # ignore in Railway, but useful locally
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
