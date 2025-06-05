from fastapi import HTTPException
from services.sugerencias_service import SuggestionService

svc = SuggestionService()

def get_suggestions(patron: list[str]):
    try:
        return svc.sugerir(patron)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
