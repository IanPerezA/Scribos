from fastapi import HTTPException
from services.sugerencias_service import SuggestionService
from utils.puntaje import complete
from utils.puntaje import score_phrase
svc = SuggestionService()

def get_suggestions(patron: list[str]):
    try:
        return svc.sugerir(patron)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

'''def get_complete_suggestion(phrase):
    try:
        words = phrase.split()
        completed_words = []

        for idx, word in enumerate(words):
            if '*' not in word:
                completed_words.append(word)
                continue

            # Reemplazar cada guion con un token [MASK]
            num_masks = word.count('*')
            mask_string = ' '.join(['[MASK]'] * num_masks)

            # Frase con palabra enmascarada
            masked_phrase = phrase.replace(word, mask_string, 1)

            # Obtener predicciones del modelo
            decoded = complete(masked_phrase)
            decoded_words = decoded.split()

            # Extraer predicciones justo en la posición esperada
            predicted_tokens = decoded_words[idx:idx + num_masks]

            # Armar palabra predicha mezclando letras conocidas + tokens predichos
            reconstructed = ""
            pred_idx = 0
            for char in word:
                if char == '*':
                    token = predicted_tokens[pred_idx]
                    # Limpiar subtokens (como ##n)
                    token = token.replace("##", "")
                    reconstructed += token
                    pred_idx += 1
                else:
                    reconstructed += char

            completed_words.append(reconstructed)

        return " ".join(completed_words)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc'''
    
from utils.puntaje import predict_masked_word

def get_complete_suggestion(phrase: str) -> str:
    words = phrase.split()
    phrases = [""]

    for word in words:
        if "*" in word:
            # Construye patrón para el indexador (lista de chars o '')
            pattern = [c if c != "*" else "" for c in word]
            # Obtiene sugerencias de palabras que encajan en ese patrón
            suggestions = get_suggestions(pattern)

            # Expande todas las combinaciones posibles
            new_phrases = []
            for base in phrases:
                for s in suggestions:
                    new_phrases.append((base + " " + s).strip())
            phrases = new_phrases

        else:
            # Si la palabra no tiene '*', la agrego tal cual
            phrases = [(base + " " + word).strip() for base in phrases]

    # Por cada frase candidata calculo su score de probabilidad y ordeno
    scored = [(p, score_phrase(p)) for p in phrases]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Devuelvo solo la mejor (o top-N si quieres)
    return scored[0][0]