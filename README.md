<p align="center">
  <img src="https://img.shields.io/badge/SMILE-%F0%9F%A7%A0_Social_Memory_Integrated_Learning_Environment-blue?style=for-the-badge" alt="SMILE Logo">
</p>

<p align="center">
  <b>SMILE</b> — <i>Social Memory Integrated Learning Environment</i><br>
  <sub>AI-driven conversational system integrating face recognition, dialogue memory, and personalized user profiling.</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&style=flat-square">
  <img src="https://img.shields.io/badge/Ollama-LLM-green?style=flat-square">
  <img src="https://img.shields.io/badge/DeepFace-Face_Recognition-orange?style=flat-square">
  <img src="https://img.shields.io/badge/Vosk-STT-yellow?style=flat-square">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square">
</p>

---

# 🧠 SMILE — *Social Memory Integrated Learning Environment*

**Version:** 1.0  
**Author:** Lorenzo (AI Robotics Lab)  
**Language:** Python 3.11  
**License:** MIT  

---

## 📘 Overview

**SMILE** (*Social Memory Integrated Learning Environment*) is an intelligent conversational agent that integrates:
- **real-time face recognition**,  
- **context-aware dialogue**,  
- and **long-term personalized memory**.

The system recognizes users visually, maintains short-term conversation history, and automatically summarizes and updates each user's **long-term profile** through interaction.

It is built for **embodied AI** and **social robotics** contexts, where continuity and personalization in human-robot dialogue are key.

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
├── data/ # Persistent user data
│ ├── known_faces/ # Registered user images
│ ├── conversations/ # Conversation transcripts
│ ├── profiles/ # User profiles (JSON)
│ └── embeddings.pkl # Face embeddings database
│
├── src/ # Source code
│ ├── config.py # Local configuration (ignored by Git)
│ ├── recognize_live.py # Main live recognition and dialogue loop
│ ├── utils/ # Functional modules
│ │ ├── dialog_manager.py
│ │ ├── memory_manager.py
│ │ ├── profile_manager.py
│ │ ├── speech_utils.py
│ │ └── facenet_utils.py
│ └── init.py
│
├── requirements.txt
└── README.md
```

---

## 🧩 Core Modules

| Module | Description |
|---------|-------------|
| **recognize_live.py** | Main entry point. Handles video stream, voice input, and conversation logic. |
| **dialog_manager.py** | Builds prompts and manages dialogue state (GREETING, FREE_TALK, FAREWELL). |
| **memory_manager.py** | Maintains working and long-term memory, generates summaries via LLM. |
| **profile_manager.py** | Handles user profiles (creation, update, persistence). |
| **speech_utils.py** | Controls TTS and STT pipelines. |
| **facenet_utils.py** | Provides facial embedding and recognition functionality. |

---

## 🚀 Getting Started

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

### 3️⃣ Configure system and Install Dependencies

```
cp src/config_example.py src/config.py
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
python -m src.recognize_live
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

When the same person is detected again, the assistant automatically loads their identity and past interactions, resuming the context seamlessly. SMILE uses a hybrid memory system:

- **Short-term memory**: the last 7 conversational exchanges

- **Long-term memory**: summarized user profile stored in `data/profiles/`

Each session enriches the user profile with new insights extracted from conversation using an LLM-based summarization pipeline.

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

## 🏗️ Future Improvements

| Milestone	| Status| Description |
|-----------|-------|-------------|
| v1.0 — SMILE Core	| ✅ Done | Real-time recognition, memory, summarization |
| v1.1 — Questionnaire Init	| 🔜 Planned | Profile initialization for new users |
| v1.2 — Emotional Context | ⏳ In design | Affective state tracking & adaptive responses |
| v2.0 — Multi-agent Setup | ⚙️ Future	| Multi-person recognition and shared context |

---

## ⚠️ Disclaimer

This system is intended for research and development in social robotics and conversational AI.
All personal data is stored locally and should be handled according to GDPR and privacy best practices.

---

## 🧑‍💻 Author

**Lorenzo D'Errico**  
PhD student in AI @ Federico II  
Email: [lorenzo.derrico@unina.it]  
LinkedIn: [linkedin/lo_de06]

---

## 🪪 License

MIT License © 2025 — Developed by Lorenzo D'Errico