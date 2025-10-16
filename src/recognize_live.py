# ==========================================
# 🎥 FACE RECOGNITION LIVE (ASYNCHRONOUS)
# ==========================================

import os
import cv2
import pickle
import time
import threading
import msvcrt
import queue

# === UTILS ===

from utils.facenet_utils import compare_embeddings
from utils.speech_utils import speak, transcribe_audio
from utils.dialog_manager import ask_ollama
from utils.async_core import (
    detect_request_q, detect_result_q,
    embed_request_q, embed_result_q,
    tts_q, start_workers, exit_event,
    speak_async, shutdown_tts_executor,
    worker_ready_event
)

# ==========================================
# ⚙️ CONFIGURAZIONE
# ==========================================

EMB_FILE = "../data/embeddings.pkl"

# ==========================================
# ⌨️ ASCOLTO TASTO 'Q'
# ==========================================

def key_listener():
    """Thread per intercettare il tasto Q in qualsiasi momento."""
    while not exit_event.is_set():
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key in [b'q', b'Q']:
                print("\n👋 Chiusura richiesta (tasto Q)...")
                exit_event.set()
                break

# ==========================================
# 🔊 INTERAZIONE (TTS + STT + LLM)
# ==========================================

def handle_interaction(name: str):
    try:
        greeting = f"Ciao {name}!" if name != "Volto rilevato" else "Ciao, piacere di conoscerti!"

        # 🔹 1. parla (blocca solo questo thread)
        future = speak_async(speak, greeting)
        future.result()  # aspetta che finisca il saluto

        # 🔹 2. breve pausa per evitare di catturare l'audio del TTS
        time.sleep(0.6)

        # 🔹 3. ascolta ora
        user_text = transcribe_audio(duration=3, stop_on_silence=True).strip()
        if not user_text:
            return

        # 🔹 4. elabora con Ollama in background
        reply_future = ask_ollama(user_text)
        reply = reply_future.result(timeout=10)

        # 🔹 5. parla la risposta (non blocca)
        speak_async(speak, reply)

    except Exception as e:
        print(f"[INTERACT] Errore: {e}")

# ==========================================
# 📦 DATABASE VOLTI
# ==========================================

def load_known_faces():
    """Carica il database di embedding noti."""
    if os.path.exists(EMB_FILE):
        with open(EMB_FILE, "rb") as f:
            known = pickle.load(f)
        print(f"✅ Caricati {len(known)} volti noti.")
        return known
    else:
        print("⚠️ Nessun volto registrato. Avvio in modalità rilevazione.")
        return {}

# ==========================================
# 🧠 MAIN LOOP
# ==========================================

def main():
    # === AVVIO WORKER E TRACKER ===
    # Avvia worker asincroni
    start_workers(speak_func=speak)

    print("🔊 Warm-up TTS...")
    speak(" ")  # una parola vuota, inizializza engine
    print("✅ TTS pronto.")

    print("🕐 Attendo che il detection worker completi il warm-up...")
    worker_ready_event.wait()
    print("✅ Detection worker pronto. Avvio webcam.")

    # Carica database
    known_faces = load_known_faces()

    # Avvia webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Errore: impossibile aprire la webcam.")
        return

    print("\n🎬 Avvio riconoscimento live...")
    print("Premi 'q' per uscire.\n")

    seen_names = set()
    trackers = {}            # id → tracker
    track_lost = {}          # id → contatore frame persi
    next_face_id = 0
    TRACKER_MAX_LOST = 8
    frame_id = 0

    # Avvia thread per ascolto tasto 'q'
    threading.Thread(target=key_listener, daemon=True).start()

    print("\n🎬 Sistema pronto. Avvio stream video...\n")

    while not exit_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame non letto correttamente.")
            break

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_id += 1

        # --- 🔹 Invia frame al detection worker (max 1 alla volta)
        if detect_request_q.qsize() < 1:
            try:
                detect_request_q.put_nowait((frame_id, rgb.copy()))
            except queue.Full:
                pass

        # --- 🔹 Recupera eventuali risultati del detection worker
        boxes = None
        try:
            while not detect_result_q.empty():
                fid, result = detect_result_q.get_nowait()
                detect_result_q.task_done()
                boxes = result
                #if boxes is not None and len(boxes) > 0:
                #    print(f"[MAIN] Ricevute {len(boxes)} box dal detection worker")
                #else:
                #    print("[MAIN] Nessuna box ricevuta dal detection worker")
        except queue.Empty:
            pass

        # --- 🔹 Se nessuna detection, aggiorna tracker esistenti
        if boxes is None and trackers:
            boxes = []
            for fid, tr in list(trackers.items()):
                ok, box = tr.update(frame)
                if ok:
                    x, y, w, h = map(int, box)
                    boxes.append([x, y, x + w, y + h])
                    track_lost[fid] = 0
                else:
                    track_lost[fid] += 1
                    if track_lost[fid] > TRACKER_MAX_LOST:
                        del trackers[fid]
                        del track_lost[fid]

        # --- 🔹 Se abbiamo nuove detection → reset tracker
        if boxes is not None:
            boxes = [b for b in boxes if b is not None]
            trackers.clear()
            track_lost.clear()
            for b in boxes:
                x1, y1, x2, y2 = map(int, b)
                w, h = x2 - x1, y2 - y1
                tracker = (
                    cv2.legacy.TrackerCSRT_create()
                    if hasattr(cv2.legacy, "TrackerCSRT_create")
                    else cv2.TrackerCSRT_create()
                )
                trackers[f"t{next_face_id}"] = tracker
                tracker.init(frame, (x1, y1, w, h))
                track_lost[f"t{next_face_id}"] = 0
                next_face_id += 1

        # --- 🔹 Disegna box e invia embedding request
        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                face_id = f"face_{i}"
                if not embed_request_q.full():
                    try:
                        embed_request_q.put_nowait((face_id, rgb.copy(), box))
                        #print(f"[EMBED-REQ] Enqueued embedding for {face_id}")
                    except queue.Full:
                        pass

        # --- 🔹 Legge eventuali embedding pronti
        try:
            while not embed_result_q.empty():
                print("[MAIN] Checking embedding results…")
                fid, embedding = embed_result_q.get_nowait()
                print(f"[MAIN] Got embedding for {fid}")
                embed_result_q.task_done()

                name = "Volto rilevato"

                # 🔍 Confronto con database volti noti
                if known_faces:
                    for person, emb_db in known_faces.items():
                        match, dist = compare_embeddings(embedding, emb_db)
                        if match:
                            name = person
                            break

                # 🔊 Se volto nuovo, avvia interazione vocale
                if name not in seen_names:
                    seen_names.add(name)
                    print(f"👁️  Nuovo volto rilevato: {name}")
                    threading.Thread(target=handle_interaction, args=(name,), daemon=True).start()

            # fine while
        except queue.Empty:
            pass


        # --- 🔹 Mostra frame
        cv2.imshow("Face Recognition Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit_event.set()
            break

    # --- 🔹 Cleanup finale
    shutdown_tts_executor()
    cap.release()
    cv2.destroyAllWindows()
    exit_event.set()
    print("\n✅ Chiusura completata.")


# ==========================================
# 🚀 AVVIO
# ==========================================
if __name__ == "__main__":
    main()