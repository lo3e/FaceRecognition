import pyaudio
import audioop
import wave
import time
import json
import os
import re
from vosk import Model, KaldiRecognizer
import pyttsx3

# ================== CONFIGURAZIONE MODELLO VOSK ==================
EXTERNAL_MODEL_DIR = r"C:\Users\brain\Documents\Universita\Erasmus\Proggetto\Dati\vosk-model-it-0.22"

def speak(text):
    engine = pyttsx3.init() 
    engine.setProperty('rate', 180) # velocità voce 
    engine.setProperty('volume', 0.9) # volume 
    voices = engine.getProperty('voices') 
    engine.setProperty('voice', voices[0].id) # 0=maschio, 1=femmina (di solito)
    engine.say(text) 
    engine.runAndWait()

# MODEL_PATH deve già essere definito come fai ora
model = Model(EXTERNAL_MODEL_DIR)
# ==========================================================

def find_working_mic(trials_rates=(16000, 48000), trials_channels=(1, 2), timeout=1.0):
    """
    Prova a trovare automaticamente un device audio apribile.
    Ritorna (device_index, rate, channels) o (None, None, None).
    """
    p = pyaudio.PyAudio()
    candidate = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] <= 0:
            continue
        name = info["name"]
        for rate in trials_rates:
            for ch in trials_channels:
                try:
                    stream = p.open(format=pyaudio.paInt16,
                                    channels=ch,
                                    rate=rate,
                                    input=True,
                                    input_device_index=i,
                                    frames_per_buffer=1024)
                    # prova a leggere un piccolo chunk
                    try:
                        data = stream.read(1024, exception_on_overflow=False)
                        if data and len(data) > 0:
                            stream.stop_stream()
                            stream.close()
                            p.terminate()
                            return i, rate, ch
                    except Exception:
                        # non valido per questa combinazione
                        stream.stop_stream()
                        stream.close()
                except Exception:
                    pass
    p.terminate()
    return None, None, None


def _to_mono_and_resample(raw_bytes, width, in_channels, in_rate, out_rate=16000):
    """
    Converte raw PCM bytes con 'width' byte/sample e in_channels in:
      - mono
      - sample rate = out_rate
    Ritorna bytes int16 a out_rate, mono.
    Usa audioop.tomono e audioop.ratecv (in-place streaming compatible).
    """
    # se stereo -> tomono (usa metà mix)
    if in_channels == 2:
        mono = audioop.tomono(raw_bytes, width, 0.5, 0.5)
    elif in_channels == 1:
        mono = raw_bytes
    else:
        # per >2 canali: prendi i primi 2 canali e mixa (semplice e robusto)
        # riduciamo a 2 canali interpretando come interleaved int16 e prendendo i primi due
        # fallback: mix byte-wise - non ideale ma evita crash
        try:
            # tentativo: converti in stereo prendendo primi due canali
            # costruzione iterativa: split frames into channels is complex; do a simple downmix by averaging samples
            mono = audioop.tomono(raw_bytes, width, 1.0 / in_channels, 1.0 / in_channels)
        except Exception:
            mono = raw_bytes  # fallback brutale
    # se bisogno ricampionare
    if in_rate != out_rate:
        try:
            state = None
            converted, state = audioop.ratecv(mono, width, 1, in_rate, out_rate, None)
            return converted
        except Exception:
            # se ratecv fallisce, ritorna input grezzo (Vosk potrebbe comunque produrre qualcosa)
            return mono
    else:
        return mono

def transcribe_audio(duration=20, stop_on_silence=True, silence_limit=1.5, silence_hangover=2.2):
    """
    Registrazione robusta:
     - durata massima `duration` secondi
     - se stop_on_silence=True aspetta silence_hangover secondi DI SILENZIO DOPO L'ULTIMO PARLATO rilevato
     - ritorna stringa trascritta
    """
    dev_idx, dev_rate, dev_ch = find_working_mic()
    if dev_idx is None:
        print("❌ [MIC] Nessun microfono accessibile trovato.")
        return ""

    RATE = dev_rate
    CHANNELS = dev_ch
    MIC_INDEX = dev_idx
    CHUNK = 2048
    FORMAT = pyaudio.paInt16

    #print(f"🎙️ [MIC] Apertura microfono index={MIC_INDEX} ({CHANNELS} ch @ {RATE}Hz)")
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=MIC_INDEX,
                        frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"⚠️ [MIC] Impossibile aprire stream (index={MIC_INDEX}): {e}")
        p.terminate()
        return ""

    frames = []
    silent_chunks = 0
    last_voice_time = None
    speech_detected = False
    start_time = time.time()
    #print("🎙️ In ascolto... (parla ora)")

    try:
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except Exception as e:
                # Read error: riprova brevemente
                time.sleep(0.01)
                continue

            if not data:
                break

            frames.append(data)
            try:
                rms = audioop.rms(data, 2)
            except Exception:
                rms = 0

            # debug minimale
            print(f"[MIC] RMS={rms}", end="\r")

            # rilevazione parlato minima
            if rms > 80:   # valore empirico: abbassalo o alzalo se necessario
                speech_detected = True
                last_voice_time = time.time()
                silent_chunks = 0
            else:
                # se abbiamo già avuto parlato, contiamo il silenzio come hangover
                if speech_detected:
                    # se sono passati silence_hangover secondi dall'ultimo parlato -> stop
                    if last_voice_time and (time.time() - last_voice_time) > silence_hangover:
                        print("\n🔇 [MIC] Silence hangover rilevato, chiusura microfono.")
                        break
                else:
                    # se non abbiamo ancora sentito nulla e siamo oltre il limite, stop per timeout
                    if time.time() - start_time > duration:
                        print("\n⏱️ [MIC] Tempo massimo raggiunto, chiusura microfono.")
                        break

            # stop globale max duration
            if time.time() - start_time > duration:
                print("\n⏱️ [MIC] Tempo massimo raggiunto, chiusura microfono.")
                break

    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        p.terminate()

    #print("\n✅ [MIC] Microfono chiuso. Elaborazione...")

    raw = b"".join(frames)
    try:
        mono16 = _to_mono_and_resample(raw, 2, CHANNELS, RATE, out_rate=16000)
    except Exception as e:
        print(f"⚠️ [MIC] Errore conversione audio: {e}")
        mono16 = raw

    # trascrizione (streaming)
    try:
        rec = KaldiRecognizer(model, 16000)
        offset = 0
        step = 4000
        text = ""
        while offset < len(mono16):
            chunk = mono16[offset:offset + step]
            if rec.AcceptWaveform(chunk):
                res = json.loads(rec.Result())
                text += " " + res.get("text", "")
            offset += step
        text += " " + json.loads(rec.FinalResult()).get("text", "")
        text = text.strip()
    except Exception as e:
        print(f"⚠️ [STT] Errore Vosk: {e}")
        text = ""

    #print(f'🗣️ [STT] Hai detto: "{text}"')
    return text

def extract_name_from_text(text: str) -> str:
    text = text.lower().strip()
    blacklist = {"piacere", "ciao", "mi", "chiamo", "sono", "eh", "io", "il", "la", "un", "una"}

    # normalizza apostrofi e punteggiatura
    text = re.sub(r"[^a-zàèéìòù\s]", "", text)

    # trova pattern "mi chiamo X" o "sono X"
    m = re.search(r"(?:mi chiamo|sono)\s+([a-zàèéìòù]+)", text)
    if m:
        name = m.group(1).capitalize()
    else:
        # prendi la parola più probabile (non nel blacklist)
        words = [w for w in text.split() if w not in blacklist]
        if not words:
            return f"Utente_{int(time.time())}"

        # se ci sono più parole, scarta le ultime tipo "piacere"
        candidates = [w for w in words if w not in ["piacere", "grazie", "ciao"]]
        name = candidates[0].capitalize()

    # evita che prenda parole generiche tipo "suo" o "bene"
    if len(name) < 3 or name in ["bene", "grazie", "suo", "ok"]:
        name = f"Utente_{int(time.time())}"

    return name
