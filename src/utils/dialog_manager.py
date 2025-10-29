import requests
import json
from utils.profile_manager import (
    load_profile,
    load_recent_history,
    format_profile_for_prompt,
    format_history_for_prompt,
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

def build_llm_prompt(user_name: str, user_text: str) -> str:
    """
    Costruisce il prompt da mandare a Ollama con:
    - profilo long-term dell'utente
    - ultimi turni di conversazione
    - istruzioni di stile
    - input corrente dell'utente
    """

    profile = load_profile(user_name)
    history = load_recent_history(user_name, window=7)

    profile_txt = format_profile_for_prompt(profile)
    history_txt = format_history_for_prompt(history)

    prompt = f"""
Sei un assistente robotico che parla in italiano, tono caldo e naturale.
Stai parlando in tempo reale con una persona che conosci fisicamente.
Il tuo ruolo è essere amichevole, presente e coerente nel tempo.

REGOLE IMPORTANTISSIME DI STILE:
- NON iniziare ogni risposta con "Ciao" o con il nome della persona.
- Saluta per nome solo all'inizio della PRIMA interazione, non ad ogni turno.
- Rispondi in modo breve, conversazionale, massimo 2-3 frasi alla volta.
- Se l'utente fa domande personali tipo "come stai?", rispondi in modo leggero, tipo un compagno di chiacchiere.
- Se l'utente ti saluta in modo di chiusura (tipo "ok allora ci sentiamo, ciao"), rispondi salutando e chiudi gentilmente.
- NON fare domande troppo dirette tutte insieme. Una domanda alla volta va bene.
- Se l'utente dice solo "ciao" o "ehi", interpretalo come inizio conversazione, NON come fine.
- Non ripetere saluti o formule di apertura (“Ciao”, “Buongiorno”, “Salve”) a ogni turno.
- Considera che la conversazione è già iniziata: rispondi direttamente al contenuto dell’utente.
- Se l’utente saluta a inizio turno, rispondi con una frase di apertura naturale, *una sola volta*, poi continua senza ripetere saluti.

Esempio:
Utente: Ciao, come va?
Assistente: Bene! È bello rivederti. Che cosa ti ha portato oggi da me?
Utente: Nulla di particolare, solo una chiacchierata.
Assistente: Ah, ottimo! Hai avuto una giornata tranquilla?

DATI SULLA PERSONA (memoria a lungo termine):
{profile_txt}

ULTIMI SCAMBI CON QUESTA PERSONA (memoria a breve termine):
{history_txt}

NUOVO INPUT DELL'UTENTE:
Utente: {user_text}

Ora rispondi come "Robot", seguendo tutte le regole sopra.
Robot:
""".strip()

    return prompt


def ask_ollama(prompt: str, model: str = MODEL_NAME) -> str:
    """
    Chiamata raw: quello che gli passi viene usato "as is".
    """
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=data)
    response.raise_for_status()
    result = response.json()
    return result.get("response", "")

def ask_ollama_with_context(user_name: str, user_text: str) -> str:
    """
    Versione high-level:
    - costruisce prompt completo con profilo + memoria breve
    - chiama Ollama
    """
    full_prompt = build_llm_prompt(user_name, user_text)
    reply = ask_ollama(full_prompt)
    return reply
