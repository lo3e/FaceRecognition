import os
import json
import pickle
import time

EMB_FILE = "../data/embeddings.pkl"
CONV_DIR = "../data/conversations"

os.makedirs(CONV_DIR, exist_ok=True)

# ====== GESTIONE VOLTI ======

def save_new_face(name, embedding):
    """Salva un nuovo volto nel database embeddings.pkl."""
    if os.path.exists(EMB_FILE):
        with open(EMB_FILE, "rb") as f:
            known = pickle.load(f)
    else:
        known = {}

    known[name] = embedding
    with open(EMB_FILE, "wb") as f:
        pickle.dump(known, f)
    print(f"💾 Nuovo volto salvato come '{name}' in embeddings.pkl")


# ====== GESTIONE CONVERSAZIONI ======

def append_conversation(name, user_text, bot_reply):
    """Salva ogni scambio in un file JSON per utente."""
    path = os.path.join(CONV_DIR, f"{name}.json")
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user": user_text,
        "bot": bot_reply
    }

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
