import os
import pyttsx3
import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer

# Percorso assoluto, indipendente da dove lanci lo script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../models/vosk-model-small-it-0.22"))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Percorso modello non trovato: {MODEL_PATH}")

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)       # velocità voce
    engine.setProperty('volume', 0.9)     # volume
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  # 0=maschio, 1=femmina (di solito)
    engine.say(text)
    engine.runAndWait()

# Inizializza modello Vosk
model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, 16000)

def transcribe_audio(duration=5):
    """Ascolta per X secondi e restituisce testo."""
    print("🎙️ In ascolto...")
    audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
    sd.wait()
    rec.AcceptWaveform(audio.tobytes())
    result = json.loads(rec.Result())
    text = result.get("text", "")
    print(f"🗣️ Hai detto: {text}")
    return text