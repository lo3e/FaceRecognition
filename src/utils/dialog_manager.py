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

def build_llm_prompt(user_name: str, user_text: str, is_first_turn: bool = False, state: str = "FREE_TALK") -> str:
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

    # --- contesto dinamico
    if state == "GREETING":
        stage = "È l'inizio della conversazione. Puoi salutare brevemente e introdurti con naturalezza."
    elif state == "FAREWELL":
        stage = "La conversazione sta per terminare. Rispondi con un saluto finale caldo e coerente, non riaprire la conversazione."
    else:
        stage = "La conversazione è già in corso. NON salutare, NON introdurre nuovamente l'utente."

    prompt = f"""
Tu sei "Robot", un assistente robotico che parla in italiano, tono caldo e naturale.
Stai parlando in tempo reale con una persona che conosci fisicamente di nome {user_name}.
Il tuo ruolo è essere amichevole, coerente e ricordare le interazioni passate.

STATO ATTUALE DEL DIALOGO: {state.upper()}
{stage}

⚠️ LINEE GUIDA IMPORTANTI ⚠️
- NON iniziare ogni risposta con "Ciao" o il nome della persona, tranne nel primissimo turno.
- Se lo stato è FREE_TALK, ignora ogni "ciao" o "buongiorno" come semplice continuazione della conversazione.
- Se lo stato è FAREWELL, rispondi con un saluto finale coerente, breve e affettuoso. Non fare domande.
- Mantieni risposte brevi (2–3 frasi), tono naturale, come una persona che ti conosce.
- Se l'utente fa domande personali tipo "come stai?", rispondi con una breve frase empatica.
- Non iniziare con saluti se non è lo stato GREETING.
- Evita di ripetere schemi ("sto qui per aiutarti", "supportarti nel tuo percorso") troppo spesso.

MEMORIA A LUNGO TERMINE (profilo utente):
{profile_txt}

MEMORIA A BREVE TERMINE (ultimi 7 scambi):
{history_txt}

UTENTE: {user_text}

Ora rispondi in modo naturale e coerente, come "Robot".
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

def ask_ollama_with_context(user_name: str, user_text: str, is_first_turn: bool = False, state: str = "FREE_TALK") -> str:
    prompt = build_llm_prompt(user_name, user_text, is_first_turn=is_first_turn, state=state)
    reply = ask_ollama(prompt)
    return reply
