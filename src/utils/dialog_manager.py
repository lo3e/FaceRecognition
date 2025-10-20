import requests
import json

def ask_ollama(prompt, model="llama3"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3",
        "prompt": f"Rispondi in italiano in modo colloquiale. L'utente ha detto: {prompt}", 
        "stream": False
        }
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    return result.get("response", "")

if __name__ == "__main__":
    user = input("Tu: ")
    answer = ask_ollama(user)
    print("LLM:", answer)
