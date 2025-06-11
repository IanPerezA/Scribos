from fastapi import HTTPException
from services.sugerencias_service import SuggestionService
from utils.puntaje import complete
svc = SuggestionService()

def get_suggestions(patron: list[str]):
    try:
        return svc.sugerir(patron)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

def get_complete_suggestion(phrase):
    try:
        words = phrase.split()
        completed_words = []

        for idx, word in enumerate(words):
            if '-' not in word:
                completed_words.append(word)
                continue

            # Reemplazar cada guion con un token [MASK]
            num_masks = word.count('-')
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
                if char == '-':
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
        raise HTTPException(status_code=500, detail=str(exc)) from exc
