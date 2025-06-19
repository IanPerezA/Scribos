# utils/puntaje.py
import torch
from utils.cargador_modelo import get_model

# Cargar modelo y tokenizer una sola vez (usando caché)
tokenizer, model = get_model()
model.eval()

def score_phrase(phrase: str) -> float:
    """Devuelve la log-probabilidad total de una frase"""
    inputs = tokenizer(phrase, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    log_probs = 0.0
    for i in range(1, input_ids.size(1)):
        token_logits = logits[0, i - 1]
        token_id = input_ids[0, i]
        prob = torch.softmax(token_logits, dim=-1)[token_id].item()
        log_probs += torch.log(torch.tensor(prob))

    return float(log_probs)

def complete(masked_phrase: str) -> str:
    """Completa una frase con uno o varios tokens [MASK]"""
    inputs = tokenizer(masked_phrase, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    mask_token_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    predicted_tokens = inputs.input_ids[0].tolist()
    for idx in mask_token_indices:
        predicted_index = torch.argmax(logits[0, idx]).item()
        predicted_tokens[idx] = predicted_index

    decoded_sentence = tokenizer.decode(predicted_tokens, skip_special_tokens=True)
    return decoded_sentence

def match_pattern(word: str, pattern: str) -> bool:
    """Verifica si una palabra coincide con un patrón tipo a*t*"""
    if len(word) != len(pattern):
        return False
    return all(pc == "*" or pc == wc for pc, wc in zip(pattern, word))

def predict_masked_word(masked_text: str, original_pattern: str, top_k=20) -> list[str]:
    top_k = int(top_k)

    inputs = tokenizer(masked_text, return_tensors="pt")
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    mask_token_logits = logits[0, mask_token_index, :]
    top_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices[0].tolist()

    candidates = [tokenizer.decode([t]).strip() for t in top_tokens]
    filtered = [w for w in candidates if match_pattern(w, original_pattern)]
    return filtered
