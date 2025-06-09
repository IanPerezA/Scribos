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
