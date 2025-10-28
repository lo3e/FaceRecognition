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
    Costruisce il prompt da mandare a Ollama:
    - Profilo long-term dell'utente
    - Ultimi turni di conversazione (working memory ~7 turni)
    - L'input corrente dell'utente
    - Istruzioni su stile (parla in italiano, tono naturale, amichevole)
    """

    profile = load_profile(user_name)
    history = load_recent_history(user_name, window=7)

    profile_txt = format_profile_for_prompt(profile)
    history_txt = format_history_for_prompt(history)

    prompt = f"""
Sei un assistente sociale robotico che parla in italiano in modo colloquiale e caldo.
Il tuo compito è ricordare le persone e parlare in modo personalizzato.

DATI SULLA PERSONA (memoria a lungo termine):
{profile_txt}

ULTIMI SCAMBI CON QUESTA PERSONA (memoria a breve termine):
{history_txt}

ISTRUZIONI DI COMPORTAMENTO:
- Rispondi in modo breve e naturale.
- Non fare domande troppo invasive tutte insieme.
- Se l'utente saluta o vuole andare via, saluta gentilmente e concludi.
- Se l'utente chiede qualcosa di tecnico, prova a rispondere con semplicità.

ORA NUOVO INPUT DELL'UTENTE:
Utente: {user_text}

Rispondi come "Robot":
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
