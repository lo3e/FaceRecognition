import os
import cv2
import pickle
import torch
import time
from facenet_pytorch import MTCNN
from deepface import DeepFace
from utils.facenet_utils import get_face_embedding_from_frame, compare_embeddings
from utils.speech_utils import speak  # nuovo modulo TTS

# Percorso file embeddings
EMB_FILE = "../data/embeddings.pkl"

# Carica database volti noti (se esiste)
if os.path.exists(EMB_FILE):
    with open(EMB_FILE, "rb") as f:
        known_faces = pickle.load(f)
    print(f"✅ Caricati {len(known_faces)} volti noti.")
else:
    known_faces = {}
    print("⚠️ Nessun volto registrato. Avvio in modalità sola rilevazione...")

# Inizializza FaceNet detector
mtcnn = MTCNN(keep_all=True)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Errore: impossibile aprire la webcam.")
    exit()

print("\n🎥 Avvio riconoscimento live...")
print("Premi 'q' per uscire.\n")

seen_names = set()
last_age_gender = {}
last_analysis_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame non letto correttamente dalla webcam.")
        break

    # Riduci risoluzione per performance migliori
    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb)

    if boxes is not None:
        for i, box in enumerate(boxes):
            try:
                x1, y1, x2, y2 = [int(b) for b in box]
                # Evita errori se la faccia esce dallo schermo
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                # Ottieni embedding sicuro
                embedding = get_face_embedding_from_frame(rgb, box)
                name = "Volto rilevato"

                # Confronto con database se disponibile
                if embedding is not None and known_faces:
                    for person, emb_db in known_faces.items():
                        match, dist = compare_embeddings(embedding, emb_db)
                        if match:
                            name = person
                            break

                # 🔹 Se è la prima volta che vediamo questa persona → parla
                if name not in seen_names:
                    greeting = f"Ciao {name}!" if name != "Volto rilevato" else "Ciao, piacere di conoscerti!"
                    speak(greeting)
                    seen_names.add(name)

                # 🔹 Analizza età/genere ogni 2 secondi (non ogni frame)
                now = time.time()
                if (now - last_analysis_time > 2) or (i not in last_age_gender):
                    try:
                        analysis = DeepFace.analyze(
                            rgb[y1:y2, x1:x2],
                            actions=["age", "gender"],
                            enforce_detection=False,
                            silent=True
                        )
                        age = analysis[0]["age"]
                        gender = analysis[0]["dominant_gender"]
                        last_age_gender[i] = (age, gender)
                        last_analysis_time = now
                    except Exception:
                        last_age_gender[i] = ("?", "?")

                age, gender = last_age_gender.get(i, ("?", "?"))

                # Disegna box e info
                label = f"{name} ({gender}, {age})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            except Exception as e:
                # Evita crash su errori singoli (es. volto troppo vicino)
                print(f"⚠️ Errore su volto {i}: {e}")
                continue

    cv2.imshow("Face Recognition Live", frame)

    # 🔹 Esci con 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n👋 Terminazione richiesta, chiusura...")
        break

cap.release()
cv2.destroyAllWindows()
