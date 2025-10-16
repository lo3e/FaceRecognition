import os
import pyttsx3
import sounddevice as sd
import json
import pyaudio
import wave
import audioop
import time
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

'''def transcribe_audio(duration=5):
    """Ascolta per X secondi e restituisce testo."""
    print("🎙️ In ascolto...")
    audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
    sd.wait()
    rec.AcceptWaveform(audio.tobytes())
    result = json.loads(rec.Result())
    text = result.get("text", "")
    print(f"🗣️ Hai detto: {text}")
    return text
    '''

def transcribe_audio(duration=3, stop_on_silence=False):
    """
    Registra audio dal microfono per un massimo di `duration` secondi.
    Se stop_on_silence=True, interrompe la registrazione se non rileva parlato.
    """

    RATE = 16000
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    SILENCE_THRESHOLD = 500  # più alto = meno sensibile
    SILENCE_LIMIT = 1.2      # secondi di silenzio consecutivo per interrompere

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

    print("🎙️ In ascolto...")

    frames = []
    silent_chunks = 0
    start_time = time.time()

    while True:
        data = stream.read(CHUNK)
        frames.append(data)

        rms = audioop.rms(data, 2)
        if stop_on_silence:
            if rms < SILENCE_THRESHOLD:
                silent_chunks += 1
            else:
                silent_chunks = 0

            # interrompe se c'è troppo silenzio
            if silent_chunks * CHUNK / RATE > SILENCE_LIMIT:
                break

        if time.time() - start_time > duration:
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    # salva temporaneamente
    wf = wave.open("temp_audio.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # passa a vosk per trascrizione
    rec = KaldiRecognizer(model, RATE)
    with open("temp_audio.wav", "rb") as f:
        rec.AcceptWaveform(f.read())
    text = json.loads(rec.FinalResult()).get("text", "")
    return text.strip()