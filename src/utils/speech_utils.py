import pyaudio
import audioop
import wave
import time
import json
import os
from vosk import Model, KaldiRecognizer
import pyttsx3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../models/vosk-model-small-it-0.22"))

def speak(text):
    engine = pyttsx3.init() 
    engine.setProperty('rate', 180) # velocità voce 
    engine.setProperty('volume', 0.9) # volume 
    voices = engine.getProperty('voices') 
    engine.setProperty('voice', voices[0].id) # 0=maschio, 1=femmina (di solito)
    engine.say(text) 
    engine.runAndWait()

# MODEL_PATH deve già essere definito come fai ora
model = Model(MODEL_PATH)
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


def transcribe_audio(duration=8, stop_on_silence=True, silence_limit=1.8):
    """
    Registrazione robusta con auto-rilevamento microfono e stop sul silenzio.
    Più tollerante, evita chiusure premature.
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
    SILENCE_THRESHOLD = 400  # più alto = meno sensibile

    print(f"🎙️ [MIC] Apertura microfono index={MIC_INDEX} ({CHANNELS} ch @ {RATE}Hz)")

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
    start_time = time.time()
    print("🎙️ In ascolto... (parla ora)")

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            if not data:
                break

            rms = audioop.rms(data, 2)
            print(f"[MIC] RMS={rms}", end="\r")

            frames.append(data)

            if stop_on_silence:
                if rms < SILENCE_THRESHOLD:
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                # chiudi solo se 1.8 secondi di vero silenzio
                if (silent_chunks * CHUNK / float(RATE)) > silence_limit:
                    print("\n🔇 [MIC] Silenzio prolungato, chiusura microfono.")
                    break

            if time.time() - start_time > duration:
                print("\n⏱️ [MIC] Tempo massimo raggiunto, chiusura microfono.")
                break

            time.sleep(0.002)

    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        p.terminate()
        
    print("✅ [MIC] Microfono chiuso. Elaborazione...")

    # concatena frames e converte in mono 16k per Vosk
    raw = b"".join(frames)
    try:
        mono16 = _to_mono_and_resample(raw, 2, CHANNELS, RATE, out_rate=16000)
    except Exception as e:
        print(f"⚠️ [MIC] Errore conversione audio: {e}")
        mono16 = raw  # fallback

    # salva temporaneo (opzionale)
    tmp = "temp_audio.wav"
    try:
        wf = wave.open(tmp, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16bit
        wf.setframerate(16000)
        wf.writeframes(mono16)
        wf.close()
    except Exception as e:
        print(f"⚠️ [MIC] Errore salvataggio WAV: {e}")

    # trascrizione con Vosk (streaming)
    try:
        rec = KaldiRecognizer(model, 16000)
        # feed a pezzi per essere più reattivi
        offset = 0
        step = 4000
        text = ""
        while offset < len(mono16):
            chunk = mono16[offset:offset + step]
            if rec.AcceptWaveform(chunk):
                res = json.loads(rec.Result())
                text += " " + res.get("text", "")
            offset += step
        resf = json.loads(rec.FinalResult())
        text += " " + resf.get("text", "")
        text = text.strip()
    except Exception as e:
        print(f"⚠️ [STT] Errore Vosk: {e}")
        text = ""

    print(f'🗣️ [STT] Hai detto: "{text}"')
    return text
