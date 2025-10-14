import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)       # velocità voce
    engine.setProperty('volume', 0.9)     # volume
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  # 0=maschio, 1=femmina (di solito)
    engine.say(text)
    engine.runAndWait()
