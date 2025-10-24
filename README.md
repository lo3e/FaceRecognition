# 🤖 FaceRecognition Assistant — Real-Time Vision & Voice AI

**FaceRecognition Assistant** is a multimodal AI system that combines:
- 🎥 **Real-time face detection and recognition**
- 🗣️ **Speech-to-text (STT)** with [Vosk](https://alphacephei.com/vosk/)
- 💬 **Conversational intelligence** powered by [Ollama](https://ollama.ai/)
- 🔊 **Text-to-speech (TTS)** responses via `pyttsx3`
- 🧠 **Persistent memory** for user identity and conversation history

It can recognize you, remember previous interactions, and carry on contextual conversations — completely offline (for STT/TTS) and locally integrated with Ollama for reasoning.

---

## 📸 Core Features

| Feature | Description |
|----------|-------------|
| 🧍‍♂️ **Face Recognition** | Detects and tracks multiple faces in real time via OpenCV and Facenet embeddings |
| 💾 **Identity Memory** | Saves and reloads user embeddings with a persistent `.pkl` file |
| 🗣️ **Speech Interaction** | Records and transcribes voice using Vosk STT |
| 💬 **Context-Aware Chat** | Uses Ollama (Llama 3 or any local model) to generate responses |
| 🧠 **Conversation History** | Each recognized person has a JSON log of past conversations |
| 🔊 **Speech Synthesis** | Generates natural TTS output with `pyttsx3` |
| 🧩 **Thread-safe Concurrency** | Ensures only one active interaction per face |
| 💡 **Configurable Silence Detection** | Dynamic voice/silence timing to improve conversational flow |

---

## 🧰 Tech Stack

- **Python 3.10+**
- **OpenCV** — face detection, tracking
- **Vosk** — offline STT engine
- **Ollama** — local LLM integration
- **pyttsx3** — TTS engine
- **Facenet / Dlib** — face embeddings
- **Threading & Async Queues** — concurrency handling

---

## 📦 Project Structure

```
FaceRecognition/
│
├── src/
│ ├── recognize_live.py # Main runtime script
│ ├── utils/
│ │ ├── speech_utils.py # Speech recognition + TTS
│ │ ├── facenet_utils.py # Embedding & comparison logic
│ │ ├── dialog_manager.py # Ollama integration
│ │ ├── async_core.py # Thread pools, async queues
│ │ └── ... # Other helpers
│ └── data/
│ ├── embeddings.pkl # Stored face embeddings
│ └── conversations/ # JSON logs per user
│
├── models/
│ └── vosk-model-small-it-0.22/ # (Optional local copy)
│
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/FaceRecognition.git
cd FaceRecognition
```

### 2️⃣ Create and Activate a Virtual Environment

```
python -m venv facenet
facenet\Scripts\activate   # Windows
# or
source facenet/bin/activate  # macOS/Linux
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Download the Vosk Model (Italian)

Download from [Vosk Models](https://alphacephei.com/vosk/models) and extract it to a local folder outside the repo (to avoid huge commits). For example:

```
C:\Users\<you>\Documents\AI\models\vosk-model-it-0.22
```

Then update the model path in:

```
EXTERNAL_MODEL_DIR = r"C:\Users\<you>\Documents\AI\models\vosk-model-it-0.22"
```
---

## 🚀 How to Run

```
cd src
python recognize_live.py
```

Once running:

- The webcam feed will open.
- When a new face is detected, it will greet you and ask your name.
- It remembers you across sessions.
- You can speak freely — it listens, transcribes, responds, and speaks back.
- Conversations are saved per user in /data/conversations/.

---

## 💾 Data Persistence

| File | Description |
|----------|-------------|
```data/embeddings.pkl``` | Stores facial embeddings and associated names |
| ```data/conversations/<user>.json``` | Conversation history for each recognized user |

---

## 🧩 Customization

- **Change language**  
  Replace the Vosk model path with another supported language model (e.g., English, Spanish, German).

- **Switch TTS voice**  
  Edit the `voices[0]` or `voices[1]` parameter in `speech_utils.py` to switch between male/female or different system voices.

- **Adjust silence detection**  
  Fine-tune `rms` thresholds, `silence_limit`, and `silence_hangover` inside the `transcribe_audio()` function for better sensitivity to pauses.

- **Change the AI model**  
  In `dialog_manager.py`, update the `"model": "llama3"` line to use a different Ollama model, such as `"mistral"`, `"llama3:instruct"`, or any locally available model.

- **Modify greetings or behavior**  
  The initial user greeting logic is defined inside `handle_interaction()` in `recognize_live.py`.  
  You can personalize how the assistant greets new or known users.

---

## 🧠 How Memory Works

Each recognized user has:
- A **face embedding** (a numeric vector representing their face)
- A **name**
- A **conversation history JSON file**

When the same person is detected again, the assistant automatically loads their identity and past interactions, resuming the context seamlessly.

---

## 🧪 Example Interaction

🧍 New face detected!
- 🤖: "Hi there! I don’t think we’ve met before — what’s your name?"
- 👤: "Hi, I’m Lorenzo."
- 🤖: "Nice to meet you, Lorenzo! I’ll remember you from now on."
...
- 🧠 [Next session]
- 🤖: "Hey Lorenzo! Welcome back. How have you been?"

---

## ⚠️ Notes

- Works best with clear audio input and good lighting conditions.
- Use a microphone configured for **16 kHz** or **48 kHz** sampling rate.
- Requires **Ollama** to be running locally (`ollama serve`).
- Do **not commit** the `models/` folder to GitHub, as the files are too large.

---

## 🧑‍💻 Author

**Lorenzo D'Errico**  
PhD student in AI @ Federico II  
Email: [lorenzo.derrico@unina.it]  
LinkedIn: [linkedin/lo_de06]

---

## 🏗️ Future Improvements

- Improve TTS voice with neural synthesis  
- Add long-term, context-aware memory per user  
- Support emotional and gesture recognition  
- Handle multi-user group conversations  
- Optimize performance and UX flow

---

## 🪪 License

MIT License © 2025 — Developed by Lorenzo D'Errico