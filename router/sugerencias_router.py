# router/sugerencias_router.py

from fastapi import APIRouter
from pydantic import BaseModel
from utils.puntaje import score_phrase
from controller.sugerencias_controller import get_suggestions, get_complete_suggestion

router = APIRouter(prefix="/suggest", tags=["suggest"])

class SuggestReq(BaseModel):
    phrase: str
    pattern: list[str]

class CompleteReq(BaseModel):
    phrase: str

@router.post("", response_model=list[str])
def suggest(req: SuggestReq):
    words = get_suggestions(req.pattern)
    phrases = [f"{req.phrase} {s}" for s in words]

    # Aquí usamos score_phrase para ordenar
    scored_phrases = [(phrase, score_phrase(phrase)) for phrase in phrases]
    scored_phrases.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored_phrases][:5]

@router.post("/complete", response_model=list[str])
def complete(req: CompleteReq):
    # Este endpoint llama a get_complete_suggestion (no usa score_phrase)
    completed = get_complete_suggestion(req.phrase)
    return [completed]
