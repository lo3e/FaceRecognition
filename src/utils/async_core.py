# src/utils/async_core.py
import threading
import queue
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from facenet_pytorch import MTCNN, InceptionResnetV1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(2)  # evita saturazione CPU se non usi GPU
print(f"async_core: using device {DEVICE}")

# ==========================================================
# 🎧 EXECUTOR PER TTS E OLLAMA
# ==========================================================

_tts_executor = None
_ollama_executor = None

# Queues
detect_request_q = queue.Queue(maxsize=2)   # main -> detection worker (push frames)
detect_result_q  = queue.Queue(maxsize=4)   # detection worker -> main (boxes)
embed_request_q  = queue.Queue(maxsize=4)   # main -> embedding worker
embed_result_q   = queue.Queue(maxsize=8)
tts_q            = queue.Queue(maxsize=8)
exit_event       = threading.Event()
embed_semaphore  = threading.Semaphore(1)   # one embedding at a time

# Initialize models here (warm-up)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

print("⚙️ GPU warm-up in corso...")

# === WARM-UP GPU (ResNet + MTCNN) ===

# 🔹 Warm-up ResNet (embedding model)
with torch.no_grad():
    dummy_tensor = torch.zeros((1, 3, 160, 160), device=DEVICE)
    _ = resnet(dummy_tensor)

print(f"✅ GPU warm-up ResNet completato su {DEVICE}")

# ==========================================================
# 🧩 EVENTO DI SINCRONIZZAZIONE DEI WORKER
# ==========================================================

worker_ready_event = threading.Event()

# Detection worker: consumes frames (rgb) and returns boxes
def detection_worker():
    print("📸 Detection worker avviato (istanza MTCNN locale).")
    local_mtcnn = MTCNN(
        keep_all=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        min_face_size=40,
        thresholds=[0.5, 0.6, 0.7]
    )

    # 🔹 Warm-up reale su frame 480x640 (simile alla webcam)
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _ = local_mtcnn.detect(dummy_frame)
    print("✅ MTCNN warm-up completato (worker ready).")

    worker_ready_event.set()

    while not exit_event.is_set():
        try:
            fid, frame_rgb = detect_request_q.get(timeout=0.1)
        except queue.Empty:
            continue

        # Copia per sicurezza per evitare corruzione del buffer
        frame_rgb = np.ascontiguousarray(frame_rgb.copy())

        try:
            boxes, probs = local_mtcnn.detect(frame_rgb)
        except Exception as e:
            print(f"[DETECT] Errore su frame {fid}: {e}")
            boxes = None

        detect_result_q.put((fid, boxes))
        detect_request_q.task_done()

        #if boxes is not None:
        #    print(f"[DETECT] frame {fid}: trovate {len(boxes)} facce.")
        #else:
        #    print(f"[DETECT] frame {fid}: nessuna faccia trovata.")

# Embedding worker (keeps as before, but calls resnet from here)
def embedding_worker():
    while not exit_event.is_set():
        try:
            face_id, frame_rgb, box = embed_request_q.get(timeout=0.5)
            #print(f"[EMBED-WORKER] Received {face_id}")
        except queue.Empty:
            continue
        with embed_semaphore:
            try:
                x1,y1,x2,y2 = [int(v) for v in box]
                face = frame_rgb[y1:y2, x1:x2]
                face_tensor = torch.tensor(face).permute(2,0,1).unsqueeze(0).float()/255.0
                face_tensor = torch.nn.functional.interpolate(face_tensor, size=(160,160)).to(DEVICE)
                with torch.no_grad():
                    emb = resnet(face_tensor).cpu().numpy()
                embed_result_q.put((face_id, emb))
            except Exception as e:
                print("async_core: embedding_worker error:", e)
            finally:
                embed_request_q.task_done()

# TTS worker (wrap speak)
def tts_worker(speak_func):
    while not exit_event.is_set():
        try:
            text = tts_q.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            speak_func(text)
        except Exception as e:
            print("async_core: tts_worker error:", e)
        finally:
            tts_q.task_done()

def start_executors():
    """Inizializza gli executor per TTS e Ollama."""
    global _tts_executor, _ollama_executor
    if _tts_executor is None:
        _tts_executor = ThreadPoolExecutor(max_workers=1)
        print("🔊 Executor TTS avviato")
    if _ollama_executor is None:
        _ollama_executor = ThreadPoolExecutor(max_workers=1)
        print("🧠 Executor Ollama avviato")

def shutdown_executors():
    """Chiude gli executor in modo pulito."""
    global _tts_executor, _ollama_executor
    if _tts_executor:
        _tts_executor.shutdown(wait=True)
        print("🔊 Executor TTS chiuso")
        _tts_executor = None
    if _ollama_executor:
        _ollama_executor.shutdown(wait=True)
        print("🧠 Executor Ollama chiuso")
        _ollama_executor = None

def speak_async(func, *args, **kwargs):
    """Esegue la funzione TTS in background."""
    if _tts_executor is None:
        start_executors()
    return _tts_executor.submit(func, *args, **kwargs)

def ask_ollama_async(func, *args, **kwargs):
    """Esegue la chiamata a Ollama in background."""
    if _ollama_executor is None:
        start_executors()
    return _ollama_executor.submit(func, *args, **kwargs)

def start_workers(speak_func=None):
    """
    Avvia tutti i thread asincroni e gli executor necessari:
      - detection_worker (MTCNN)
      - embedding_worker (ResNet)
      - executor TTS e Ollama
    """
    # === 1️⃣ AVVIO WORKER DI RILEVAZIONE E EMBEDDING ===
    threading.Thread(target=detection_worker, daemon=True).start()
    threading.Thread(target=embedding_worker, daemon=True).start()

    # === 2️⃣ AVVIO EXECUTOR TTS + OLLAMA ===
    start_executors()   # <-- sostituisce start_tts_executor()

    print("async_core: workers started")