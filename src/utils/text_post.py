import re

def clean_llm_reply(raw_text: str, state: str, is_first_turn: bool) -> str:
    """
    Pulisce la risposta grezza dell'LLM prima del TTS:
    - rimuove saluti ripetuti tipo "Ciao!" se non è il primo turno
    - elimina boilerplate troppo lungo/stile call center
    - in fase di chiusura (FAREWELL) evita di riaprire la conversazione
    """
    text = raw_text.strip()

    # normalizza spazi multipli / newline strani
    text = re.sub(r'\s+', ' ', text)

    # 1. Se NON è il primo turno e NON siamo in GREETING,
    #    togli prefissi tipo "Ciao!", "Ciao Lorenzo!", ecc. solo all'inizio.
    if not is_first_turn and state != "GREETING":
        text = re.sub(
            r'^(ciao[,!\.]?\s*)',  # "Ciao " / "Ciao!" / "Ciao, "
            '',
            text,
            flags=re.IGNORECASE
        )
        # anche forme tipo "Ciao Lorenzo!"
        text = re.sub(
            r'^(ciao\s+[A-Za-zÀ-ÖØ-öø-ÿ]+[,!\.]?\s*)',
            '',
            text,
            flags=re.IGNORECASE
        )

    # 2. Rimuovi boilerplate ripetitivo stile "sto sempre qui per supportarti"
    repetitive_patterns = [
        r"sto sempre felice di vederti[^.]*\.", 
        r"sto sempre qui per aiutarti[^.]*\.",
        r"sono felice di poterti supportare[^.]*\.",
        r"non ti preoccupare[^.]*\.",  # spesso suona paternalistico e ripetuto
        r"qualunque cosa tu abbia bisogno[^.]*\.",
    ]
    for pat in repetitive_patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)

    # 3. Se siamo in FAREWELL:
    #    - tieni solo le prime 1-2 frasi massimo
    #    - niente domande "come stai oggi?"
    if state == "FAREWELL":
        # taglia dopo 2 frasi
        sentences = re.split(r'([.?!])', text)
        # sentences è tipo ["bla", ".", " bla", ".", " altra", "." ...]
        rebuilt = ""
        count_full = 0
        for i in range(0, len(sentences), 2):
            if i+1 < len(sentences):
                piece = (sentences[i] + sentences[i+1]).strip()
            else:
                piece = sentences[i].strip()
            if not piece:
                continue
            # scarta frasi interrogative nella chiusura tipo "Come stai oggi?"
            if '?' in piece:
                continue
            rebuilt += (" " + piece)
            count_full += 1
            if count_full >= 2:
                break
        text = rebuilt.strip()

    # pulizia finale spazi doppi rimasti
    text = re.sub(r'\s+', ' ', text).strip()

    return text
