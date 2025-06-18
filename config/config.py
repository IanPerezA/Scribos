from functools import lru_cache
from pathlib import Path
from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "Scribos"
    api_prefix:   str = "/api/v1"
    debug:        bool = False

    MODEL_NAME:      str  = "dccuchile/bert-base-spanish-wwm-cased"
    LOCAL_MODEL_DIR: Path = Path("modelo_beto")
    WORDS_FILE:      Path = Path("data/words.txt")
    INDEX_FILE:      Path = Path("data/index.json")
    DRIVE_MODEL_URL: str = "https://drive.google.com/file/d/14kk86vRe98SzPP2WqZE7wuIN2eT9whVJ"

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file="C:/Users/HP/Documents/repo-scribos/Scribos/.env",       
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache               
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
