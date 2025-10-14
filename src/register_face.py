import os
import pickle
from utils.facenet_utils import get_face_embedding

DATA_PATH = "../data"
KNOWN_FACES = os.path.join(DATA_PATH, "known_faces")
EMB_FILE = os.path.join(DATA_PATH, "embeddings.pkl")

os.makedirs(KNOWN_FACES, exist_ok=True)

name = input("Nome della persona da registrare: ").strip()
img_path = input("Percorso immagine (es. mario.jpg): ").strip()

embedding = get_face_embedding(img_path)

if embedding is not None:
    if os.path.exists(EMB_FILE):
        with open(EMB_FILE, 'rb') as f:
            database = pickle.load(f)
    else:
        database = {}

    database[name] = embedding

    with open(EMB_FILE, 'wb') as f:
        pickle.dump(database, f)

    print(f"✅ Persona '{name}' registrata con successo!")
else:
    print("❌ Nessun volto rilevato nell'immagine.")
