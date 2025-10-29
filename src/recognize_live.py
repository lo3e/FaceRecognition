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
import traceback

# === UTILS ===

from utils.facenet_utils import compare_embeddings
from utils.speech_utils import speak, transcribe_audio, extract_name_from_text
from utils.dialog_manager import ask_ollama_with_context, ask_ollama
from utils.profile_manager import update_profile_notes
from utils.memory_manager import append_conversation, save_new_face
from utils.async_core import (
    detect_request_q, detect_result_q,
    embed_request_q, embed_result_q,
    start_workers, exit_event,
    speak_async, shutdown_executors,
    worker_ready_event, ask_ollama_async,
    embedding_ready_event
)

# ==========================================
# ⚙️ CONFIGURAZIONE
# ==========================================

EMB_FILE = "../data/embeddings.pkl"

# 🔧 Configurazione ottimizzata
TRACKER_MAX_LOST = 15  # 🔧 Aumentato da 8 (più tollerante)
EMBED_INTERVAL = 20.0   # 🔧 Secondi tra embedding dello stesso tracker
RESEEN_THRESHOLD = 30  # 🔧 Secondi prima di ri-salutare
IOU_THRESHOLD = 0.3    # 🔧 Soglia IoU per matching

# ==========================================
# 🧮 UTILITY FUNCTIONS
# ==========================================

def iou(box1, box2):
    """
    Calcola Intersection over Union tra due box.
    box format: (x, y, w, h)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

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
last_interaction_time = 0
conversation_lock = threading.Lock()

'''def handle_interaction(name: str, embedding=None):
    try:
        # Se il volto è sconosciuto → chiedi il nome
        if name == "Volto rilevato" and embedding is not None:
            speak_async(speak, "Ciao! Non credo di averti mai conosciuto prima, come ti chiami?").result()
            time.sleep(1.2)

            user_name = transcribe_audio(duration=12, stop_on_silence=True, silence_limit=3.5).strip()
            name = extract_name_from_text(user_name)

            speak_async(speak, f"Piacere {name}! D'ora in poi ti riconoscerò. Cosa ci fai da queste parti?").result()
            save_new_face(name, embedding)
            time.sleep(1.2)
        else:
            speak_async(speak, f"Ciao {name}!").result()
            time.sleep(1.2)

        conversation_active = True
        silence_counter = 0
        max_silence_rounds = 3
        farewell_keywords = [
            "ciao", "ci vediamo", "alla prossima", "a presto", "arrivederci",
            "buona giornata", "buona serata", "saluto", "vado"
        ]

        print("\n🟢 Conversazione attiva — puoi parlare ora!\n")

        while conversation_active and not exit_event.is_set():
            # 🎙️ Aumentiamo il tempo per non tagliare le frasi
            user_text = transcribe_audio(duration=20, stop_on_silence=True, silence_limit=3.5, silence_hangover=3.5).strip()

            if not user_text:
                silence_counter += 1
                print(f"🤫 Silenzio rilevato ({silence_counter}/{max_silence_rounds})")
                if silence_counter >= max_silence_rounds:
                    print("🕓 Nessuna risposta per troppo tempo, termino la conversazione.")
                    break
                continue

            silence_counter = 0

            print(f"🗣️ [STT] Hai detto: \"{user_text}\"")

            # 🔍 rileva saluti
            if any(kw in user_text.lower() for kw in farewell_keywords):
                print("👋 Rilevato saluto di chiusura.")
                speak_async(speak, f"Ciao {name}, a presto!").result()
                append_conversation(name, user_text, f"Ciao {name}, a presto!")
                update_profile_notes(name, f"L'utente ha salutato dicendo: \"{user_text}\".")
                break

            # --- Genera risposta
            reply_future = ask_ollama_async(ask_ollama_with_context, name, user_text)
            reply = reply_future.result(timeout=30)

            print(f"🤖 [LLM] Risposta: \"{reply}\"")

            # --- TTS
            speak_async(speak, reply).result()
            append_conversation(name, user_text, reply)
            update_profile_notes(name, f"L'utente ha detto: \"{user_text}\". Il robot ha risposto: \"{reply}\".")
            print(f"🔊 [TTS] Ho detto: \"{reply}\"\n")

            # 🔄 breve pausa naturale
            print("🔴 In attesa... (sto ascoltando di nuovo fra poco)")
            time.sleep(1.0)
            print("🟢 Pronto ad ascoltare!")

            # 🔚 comandi manuali di chiusura
            if user_text.lower() in ["esci", "stop", "basta"]:
                print("👋 Conversazione terminata su comando vocale.")
                break

        print(f"✅ Conversazione con {name} terminata.\n")

    except Exception as e:
        print(f"[INTERACT] Errore: {e}")'''

def handle_interaction(name: str, embedding=None):
    try:
        # === 1. Saluto iniziale / riconoscimento utente ===
        # ramo: utente nuovo -> chiedi nome e registra
        if name == "Volto rilevato" and embedding is not None:
            speak_async(speak, "Ciao! Non credo di averti mai conosciuto prima, come ti chiami?").result()
            time.sleep(1.2)

            user_name = transcribe_audio(
                duration=12,
                stop_on_silence=True,
                silence_limit=3.5
            ).strip()

            # Estrai nome pulito
            name = extract_name_from_text(user_name)

            speak_async(speak, f"Piacere {name}! D'ora in poi ti riconoscerò. Dimmi pure, come va oggi?").result()
            save_new_face(name, embedding)
            time.sleep(1.2)

        else:
            # utente già noto
            speak_async(speak, f"Ciao {name}!").result()
            time.sleep(1.2)

        # === 2. Stato conversazionale ===
        # GREETING: primi turni dopo il riconoscimento
        # FREE_TALK: conversazione libera
        # FAREWELL: chiusura
        state = "GREETING"

        if state == "GREETING":
            print("👋 Stato iniziale: GREETING")

        silence_counter = 0
        max_silence_rounds = 3

        # parole che indicano un saluto iniziale
        greeting_keywords = [
            "ciao",
            "salve",
            "buongiorno",
            "buonasera",
            "hey",
            "ehi",
            "hola"
        ]
        # parole che "potrebbero" significare addio,
        # MA verranno controllate solo se siamo già in FREE_TALK
        farewell_keywords = [
            "ciao",
            "ci vediamo",
            "alla prossima",
            "a presto",
            "arrivederci",
            "buona giornata",
            "buona serata",
            "vado",
            "devo andare"
        ]

        print("\n🟢 Conversazione attiva — puoi parlare ora!\n")

        # === 3. loop conversazionale ===
        while not exit_event.is_set():
            # ascolta utente
            user_text = transcribe_audio(
                duration=20,
                stop_on_silence=True,
                silence_limit=3.2
            ).strip()

            # gestione silenzio / inattività
            if not user_text:
                silence_counter += 1
                print(f"🤫 Silenzio rilevato ({silence_counter}/{max_silence_rounds})")

                if silence_counter >= max_silence_rounds:
                    print("🕓 Nessuna risposta per troppo tempo, termino la conversazione.")
                    break

                # se stiamo ancora nel GREETING ma l'utente non parla, non passiamo a FREE_TALK
                # semplicemente continuiamo ad ascoltare
                continue

            # abbiamo parlato -> reset contatore silenzio
            silence_counter = 0
            print(f"🗣️ [STT] Hai detto: \"{user_text}\"")

            lower_text = user_text.lower()

            # === 3a. Gestione dello stato GREETING ===
            if state == "GREETING":
                # logica: nel GREETING non interpretiamo "ciao" come addio.
                # appena l'utente dice QUALCOSA di più del solo "ciao", passiamo a FREE_TALK
                # esempi validi per passare a FREE_TALK:
                # - "ciao come stai"
                # - "tutto bene?"
                # - "sto facendo un esperimento"
                if len(lower_text.split()) > 1 or "come" in lower_text or "sto" in lower_text:
                    state = "FREE_TALK"
                else:
                    # se ha detto solo "ciao", rispondi breve e resta in GREETING
                    reply_future = ask_ollama_async(
                        lambda prompt: ask_ollama_with_context(name, prompt),
                        user_text
                        )
                    reply = reply_future.result(timeout=30)

                    speak_async(speak, reply).result()
                    append_conversation(name, user_text, reply)
                    update_profile_notes(name, f"L'utente ha detto: \"{user_text}\". Il robot ha risposto: \"{reply}\".")

                    # riprendi ascolto
                    print("🟢 Pronto ad ascoltare!")
                    time.sleep(1.0)
                    continue  # torna a inizio while senza valutare free talk / chiusura


            # === 3b. Gestione saluto di chiusura (solo se già in FREE_TALK)
            # condizione: user_text è corto e contiene parole di congedo
            is_goodbye = (
                state == "FREE_TALK"
                and any(kw in lower_text for kw in farewell_keywords)
                and not any(kw in lower_text for kw in greeting_keywords)
            )

            # --- Rilevato saluto finale
            if is_goodbye:
                print("👋 Rilevato saluto di chiusura.")

                farewell_prompt = (
                    f"L'utente {name} ha detto: '{user_text}'. "
                    f"Genera un saluto di chiusura naturale, coerente e contestuale "
                    f"che mostri empatia e si congedi in modo amichevole, "
                    f"senza ripetere 'Ciao' più di una volta."
                )

                reply_future = ask_ollama_async(
                    lambda prompt: ask_ollama_with_context(name, prompt),
                    farewell_prompt
                )
                farewell_reply = reply_future.result(timeout=20)

                speak_async(speak, farewell_reply).result()
                append_conversation(name, user_text, farewell_reply)
                print(f"🔊 [TTS] Ho detto: \"{farewell_reply}\"")
                break

            # === 3c. Conversazione normale (FREE_TALK)
            # chiediamo all'LLM una risposta contestuale (profilo + memoria breve)
            reply_future = ask_ollama_async(
                lambda prompt: ask_ollama_with_context(name, prompt),
                user_text
                )
            reply = reply_future.result(timeout=30)

            print(f"🤖 [LLM] Risposta: \"{reply}\"")

            speak_async(speak, reply).result()

            append_conversation(name, user_text, reply)
            update_profile_notes(
                name,
                f"L'utente ha detto: \"{user_text}\". Il robot ha risposto: \"{reply}\"."
            )

            print(f"🔊 [TTS] Ho detto: \"{reply}\"\n")

            # piccola pausa naturale e loop continua
            time.sleep(1.0)
            print("🟢 Pronto ad ascoltare!")

        # === 4. Uscita pulita dalla conversazione ===
        print(f"✅ Conversazione con {name} terminata.\n")

        # (qui in Step 3 metteremo anche l'aggiornamento del profilo a fine chat)

    except Exception as e:
        print("[INTERACT] Errore:", repr(e))
        traceback.print_exc()

def handle_interaction_threadsafe(name, embedding=None):
    global last_interaction_time
    with conversation_lock:
        last_interaction_time = time.time()
        handle_interaction(name, embedding)
        last_interaction_time = time.time()

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
    start_workers(speak_func=speak)

    print("🔊 Warm-up TTS...")
    speak(" ")
    print("✅ TTS pronto.")

    print("🕐 Attendo che i worker completino il warm-up...")
    worker_ready_event.wait()
    embedding_ready_event.wait()
    print("✅ Tutti i worker pronti. Avvio webcam.")

    # Carica database
    known_faces = load_known_faces()

    # Avvia webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Errore: impossibile aprire la webcam.")
        return

    print("\n🎬 Avvio riconoscimento live...")
    print("Premi 'q' per uscire.\n")

    # 🔧 FIX: seen_names ora è un dict con timestamp
    seen_names = {}  # name -> timestamp ultimo saluto
    active_interactions = {}  # name/id -> thread attiva
    
    trackers = {}            # id → tracker
    track_lost = {}          # id → contatore frame persi
    tracker_boxes = {}       # id → (x, y, w, h) ultima box nota
    last_embed_time = {}     # id → timestamp ultimo embedding
    next_face_id = 0
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
        current_time = time.time()

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
                det_fid, result = detect_result_q.get_nowait()
                detect_result_q.task_done()
                boxes = result
        except queue.Empty:
            pass

        # --- 🔹 Se nessuna detection, aggiorna tracker esistenti
        if boxes is None and trackers:
            boxes = []
            for tid, tr in list(trackers.items()):
                ok, box = tr.update(frame)
                if ok:
                    x, y, w, h = map(int, box)
                    boxes.append([x, y, x + w, y + h])
                    track_lost[tid] = 0
                    tracker_boxes[tid] = (x, y, w, h)
                else:
                    track_lost[tid] += 1
                    # 🔧 FIX: Più tollerante prima di rimuovere
                    if track_lost[tid] > TRACKER_MAX_LOST:
                        print(f"❌ Tracker {tid} perso definitivamente")
                        del trackers[tid]
                        del track_lost[tid]
                        tracker_boxes.pop(tid, None)
                        last_embed_time.pop(tid, None)

        # --- 🔹 Gestione dei tracker (crea nuovi, aggiorna, rimuove persi)
        if boxes is not None:
            boxes = [b for b in boxes if b is not None]

            updated_trackers = {}
            matched_ids = set()

            for b in boxes:
                x1, y1, x2, y2 = map(int, b)
                w, h = x2 - x1, y2 - y1
                new_box = (x1, y1, w, h)

                # 🔧 FIX: Matching basato su IoU
                best_iou = IOU_THRESHOLD
                matched_id = None
                
                for tid in list(trackers.keys()):
                    if tid in matched_ids:
                        continue
                    
                    if tid in tracker_boxes:
                        current_iou = iou(new_box, tracker_boxes[tid])
                        if current_iou > best_iou:
                            best_iou = current_iou
                            matched_id = tid

                # 🔹 Se trovato match, re-inizializza tracker
                if matched_id is not None:
                    trackers[matched_id].init(frame, new_box)
                    updated_trackers[matched_id] = trackers[matched_id]
                    tracker_boxes[matched_id] = new_box
                    track_lost[matched_id] = 0
                    matched_ids.add(matched_id)
                else:
                    # 🔹 Crea nuovo tracker
                    tracker = (
                        cv2.legacy.TrackerCSRT_create()
                        if hasattr(cv2.legacy, "TrackerCSRT_create")
                        else cv2.TrackerCSRT_create()
                    )
                    tid = f"t{next_face_id}"
                    tracker.init(frame, new_box)
                    updated_trackers[tid] = tracker
                    tracker_boxes[tid] = new_box
                    track_lost[tid] = 0
                    print(f"🆕 Nuovo tracker {tid} creato {new_box}")
                    next_face_id += 1
                    matched_ids.add(tid)

            trackers = updated_trackers

        # --- 🔹 Disegna box e invia embedding request (con rate limiting)
        for tid, tr in list(trackers.items()):
            ok, box = tr.update(frame)
            
            if not ok or box is None:
                track_lost[tid] += 1
                if track_lost[tid] > TRACKER_MAX_LOST:
                    print(f"❌ Tracker {tid} perso, rimosso")
                    trackers.pop(tid, None)
                    track_lost.pop(tid, None)
                    tracker_boxes.pop(tid, None)
                    last_embed_time.pop(tid, None)
                continue
            
            x, y, w, h = map(int, box)
            
            if w <= 0 or h <= 0:
                track_lost[tid] += 1
                continue
            
            track_lost[tid] = 0
            tracker_boxes[tid] = (x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 🔧 FIX: Rate limiting embedding request
            if tid not in last_embed_time or current_time - last_embed_time[tid] > EMBED_INTERVAL:
                if not embed_request_q.full():
                    try:
                        embed_request_q.put_nowait((tid, rgb.copy(), (x, y, x + w, y + h)))
                        last_embed_time[tid] = current_time
                    except queue.Full:
                        pass

        # --- 🔹 Legge eventuali embedding pronti
        try:
            while not embed_result_q.empty():
                emb_fid, embedding = embed_result_q.get_nowait()
                embed_result_q.task_done()

                name = "Volto rilevato"

                # 🔍 Confronto con database volti noti
                if known_faces:
                    for person, emb_db in known_faces.items():
                        match, dist = compare_embeddings(embedding, emb_db)
                        if match:
                            name = person
                            break

                # === 🔧 FIX: evita doppie interazioni ===
                current_time = time.time()
                # se sconosciuto → usa id tracker come chiave unica
                display_key = name if name != "Volto rilevato" else emb_fid

                # Controlla se è già attiva un’interazione per questo volto
                existing = active_interactions.get(display_key)
                if existing and getattr(existing, "is_alive", lambda: False)():
                    # già in conversazione, aggiorna solo timestamp e salta
                    seen_names[name] = current_time
                    continue

                # Controlla cooldown per ri-saluto
                if name not in seen_names or current_time - seen_names[name] > RESEEN_THRESHOLD:
                    if conversation_lock.locked():
                        print(f"⏳ Attesa fine conversazione corrente prima di interagire con {name}.")
                    else:
                        seen_names[name] = current_time
                        print(f"👁️  Nuovo volto rilevato: {name}")

                        # Avvia nuova interazione in thread dedicato
                        th = threading.Thread(target=handle_interaction_threadsafe, args=(name, embedding), daemon=True)
                        active_interactions[display_key] = th
                        th.start()

                    # Thread watcher che rimuove la entry a fine interazione
                    def _cleanup_thread(t, key):
                        t.join()
                        active_interactions.pop(key, None)

                    threading.Thread(target=_cleanup_thread, args=(th, display_key), daemon=True).start()

        except queue.Empty:
            pass

        # --- 🔹 Mostra frame
        cv2.imshow("Face Recognition Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit_event.set()
            break

    # --- 🔹 Cleanup finale
    shutdown_executors()
    cap.release()
    cv2.destroyAllWindows()
    exit_event.set()
    print("\n✅ Chiusura completata.")


# ==========================================
# 🚀 AVVIO
# ==========================================
if __name__ == "__main__":
    main()