import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

ruta_local = "./modelo_beto"
tokenizer = AutoTokenizer.from_pretrained(ruta_local)
model = AutoModelForMaskedLM.from_pretrained(ruta_local)
model.eval()

def score_phrase(phrase: str) -> float:
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

    return log_probs.item()

def complete(masked_phrase):
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