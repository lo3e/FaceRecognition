import os
import cv2
import pickle
import torch
import time
import threading
import msvcrt
from facenet_pytorch import MTCNN, InceptionResnetV1
from deepface import DeepFace
from utils.facenet_utils import compare_embeddings
from utils.speech_utils import speak, transcribe_audio
from utils.dialog_manager import ask_ollama

# ==============================
# ⚙️ EXIT FLAG
# ==============================

exit_flag = False

def key_listener():
    global exit_flag
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key in [b'q', b'Q']:
                exit_flag = True
                break

# ==============================
# 🔊 THREAD PER INTERAZIONE
# ==============================

def handle_interaction(name):
    """Thread separato per parlare, ascoltare e rispondere."""
    try:
        greeting = f"Ciao {name}!" if name != "Volto rilevato" else "Ciao, piacere di conoscerti! Come stai?"
        speak(greeting)
        user_text = transcribe_audio(duration=5)
        if user_text.strip():
            reply = ask_ollama(user_text)
            speak(reply)
    except Exception as e:
        print(f"⚠️ Errore nel thread vocale: {e}")

# ==============================
# ⚙️ CONFIGURAZIONE
# ==============================

EMB_FILE = "../data/embeddings.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Utilizzo dispositivo: {DEVICE}")

# ==============================
# 📦 CARICAMENTO DATABASE
# ==============================

if os.path.exists(EMB_FILE):
    with open(EMB_FILE, "rb") as f:
        known_faces = pickle.load(f)
    print(f"✅ Caricati {len(known_faces)} volti noti.")
else:
    known_faces = {}
    print("⚠️ Nessun volto registrato. Avvio in modalità sola rilevazione...")

# ==============================
# 🔍 INIZIALIZZA FACENET E MTCNN
# ==============================

mtcnn = MTCNN(keep_all=True, device=DEVICE)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

# ==============================
# 🎥 AVVIO CAMERA
# ==============================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Errore: impossibile aprire la webcam.")
    exit()

print("\n🎬 Avvio riconoscimento live...")
print("Premi 'q' per uscire.\n")

seen_names = set()
last_age_gender = {}
last_analysis_time = 0
frame_count = 0

# ==============================
# 🎧 THREAD ASCOLTO TASTO 'Q'
# threading.Thread(target=key_listener, daemon=True).start()

# ==============================
# 🔁 LOOP PRINCIPALE
# ==============================

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame non letto correttamente dalla webcam.")
        break

    # Riduci risoluzione per performance migliori
    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Elabora 1 frame ogni 3 per alleggerire CPU
    frame_count += 1
    if frame_count % 3 != 0:
        cv2.imshow("Face Recognition Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    boxes, _ = mtcnn.detect(rgb)

    if boxes is not None:
        for i, box in enumerate(boxes):
            try:
                x1, y1, x2, y2 = [int(b) for b in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                # ====== FACE EMBEDDING (GPU) ======
                face = rgb[y1:y2, x1:x2]
                face_tensor = torch.tensor(face).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                face_tensor = torch.nn.functional.interpolate(face_tensor, size=(160, 160))
                face_tensor = face_tensor.to(DEVICE)

                with torch.no_grad():
                    embedding = resnet(face_tensor).cpu().numpy()

                name = "Volto rilevato"

                # ====== CONFRONTO CON DATABASE ======
                if known_faces:
                    for person, emb_db in known_faces.items():
                        match, dist = compare_embeddings(embedding, emb_db)
                        if match:
                            name = person
                            break

                # ====== NUOVO VOLTO → THREAD INTERAZIONE ======
                if name not in seen_names:
                    seen_names.add(name)
                    threading.Thread(target=handle_interaction, args=(name,), daemon=True).start()

                # ====== ANALISI ETÀ/GENERE OGNI 2s ======
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

                # ====== DISEGNA BOX ======
                label = f"{name} ({gender}, {age})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            except Exception as e:
                print(f"⚠️ Errore su volto {i}: {e}")
                continue

    cv2.imshow("Face Recognition Live", frame)

    # 🔚 Esci con 'q'
    if exit_flag:
        print("\n👋 Terminazione richiesta, chiusura...")
        break

cap.release()
cv2.destroyAllWindows()
