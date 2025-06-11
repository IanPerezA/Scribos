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

    # Calcular score BETO para cada frase
    scored_phrases = [(phrase, score_phrase(phrase)) for phrase in phrases]

    # Ordenar descendente por score (más probable primero)
    scored_phrases.sort(key=lambda x: x[1], reverse=True)

    recomendation = [p[0] for p in scored_phrases]

    return recomendation[0:5]

@router.post("/complete")
def complete(req: CompleteReq):
    words = req.phrase.split(" ")

    phrases = [""]  # Lista de frases completas

    for word in words:
        if "-" in word:
            # Crear el patrón
            pattern = [char if char != "-" else "" for char in word]
            suggestions = get_suggestions(pattern)

            # Generar nuevas combinaciones
            new_phrases = []
            for phrase in phrases:
                for suggestion in suggestions:
                    new_phrase = (phrase + " " + suggestion).strip()
                    new_phrases.append(new_phrase)
            phrases = new_phrases

        else:
            # Palabra completa: simplemente añadirla a todas las frases
            phrases = [(phrase + " " + word).strip() for phrase in phrases]

    # Calcular score BETO para cada frase
    scored_phrases = [(phrase, score_phrase(phrase)) for phrase in phrases]

    # Ordenar descendente por score (más probable primero)
    scored_phrases.sort(key=lambda x: x[1], reverse=True)

    recomendation = [p[0] for p in scored_phrases]

    return recomendation[:5]
