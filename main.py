import json
from pynput import keyboard
'''from pynput import keyboard
from transformers import AutoTokenizer, AutoModelForMaskedLM
from collections import defaultdict
import torch

ruta_local = "./modelo_beto"
tokenizer = AutoTokenizer.from_pretrained(ruta_local)
model = AutoModelForMaskedLM.from_pretrained(ruta_local)
sequence = []
word = []

# FUNCIONES
def buildIndex(words):
    index = defaultdict(lambda: defaultdict(list))

    words.sort()
    for word in words:
        for i, char in enumerate(word):
            index[i][char].append(word)
    return index

def loadIndex(filename="index.json"):
    with open(filename, "r", encoding="utf-8") as f:
        index = json.load(f)
    return index

def convert_defaultdict(obj):
    if isinstance(obj, defaultdict):
        return {k: convert_defaultdict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return obj
    else:
        return obj

def searchWord(index, pattern):
    sets = []
    for i, letter in enumerate(pattern):
        if letter != '':
            candidates = index.get(i, {}).get(letter, [])
            sets.append(set(candidates))

    if not sets:
        return []
    return list(set.intersection(*sets))

def on_press(key):
    global word, index, phrase, is_incomplete, is_chossing, phrases
    
    print(f"Phrase: {phrase}")

    if is_chossing:
        print(int(key.char))
        phrase = phrases[int(key.char)]

        word = []
        phrases = []
        is_incomplete = False
        is_chossing = False
        

    else:
        if key == keyboard.Key.esc:
            return False
        
        elif key == keyboard.Key.shift or key == keyboard.Key.shift_l:       
            if is_incomplete:
                options = searchWord(index, word)
            else:
                options = [''.join(word)]   

            phrases = [phrase + ' ' + option for option in options]
            
            if len(phrases) > 1:
                print("Opciones:")
                for i, p in enumerate(phrases):
                    print(f"{i}: {p}")
                is_chossing = True
            else:
                phrase = phrases[0]
                phrases = []
            
                is_incomplete = False
                word = []

        elif key == keyboard.Key.space:
            word.append('')
            is_incomplete = True
        else:
            try:
                word.append(key.char)
            except AttributeError:
                word.append('')


if __name__ == '__main__':
    phrase = ""
    phrases = []
    is_incomplete = False
    is_chossing = False

    index = loadIndex("index.json")
    print("readeded")
    
    for position in index:
        print(f"Posición {position}:")
        for letter in index[position]:
            print(f"  Letra '{letter}': {index[position][letter]}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    listener.join()

    print(phrase)
#lacasaesdete a1
#lacasadete a1es'''

from fastapi import FastAPI
from router.sugerencias_router import router as sugerencias_router

app = FastAPI(title="Scribos API")
app.include_router(sugerencias_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)

