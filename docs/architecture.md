# 🧠 SMILE System Architecture

**Project:** SMILE — *Social Memory Integrated Learning Environment*  
**Version:** 1.0  
**Author:** Lorenzo (AI Robotics Lab)  
**Date:** November 2025  

---

## 🏗️ Overview

SMILE integrates **computer vision**, **speech recognition**, and **large language models** to enable a robot or digital agent to recognize people, remember them across sessions, and sustain personalized dialogue over time.

The architecture is modular, built around four main layers:

1. **Perception Layer** — detects and recognizes users via facial embeddings.  
2. **Conversation Layer** — manages dialogue, prompt building, and state transitions.  
3. **Memory Layer** — maintains both short-term and long-term memory.  
4. **Profile Layer** — stores and updates structured user profiles.

---

## 🧩 System Diagram

      ┌────────────────────────────┐
      │        USER INPUT          │
      │  (Face + Voice + Speech)   │
      └────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │     PERCEPTION LAYER         │
    │ ─────────────────────────────│
    │ • Face detection (OpenCV)    │
    │ • Embedding extraction       │
    │ • User identification        │
    └────────────┬─────────────────┘
                 │
                 ▼
    ┌──────────────────────────────┐
    │   CONVERSATION LAYER         │
    │ ─────────────────────────────│
    │ • Dialogue state management  │
    │ • Contextual prompt builder  │
    │ • LLM-based response         │
    │ • Speech synthesis (TTS)     │
    └────────────┬─────────────────┘
                 │
                 ▼
    ┌──────────────────────────────┐
    │      MEMORY LAYER            │
    │ ─────────────────────────────│
    │ • Short-term history         │
    │ • Conversation logging       │
    │ • LLM summarization          │
    └────────────┬─────────────────┘
                 │
                 ▼
    ┌──────────────────────────────┐
    │      PROFILE LAYER           │
    │ ─────────────────────────────│
    │ • JSON profile persistence   │
    │ • Semantic field updates     │
    │ • Long-term learning         │
    └──────────────────────────────┘

---

## 🧠 Memory Hierarchy

| Memory Type | Persistence | Source | Content | Example |
|--------------|-------------|---------|----------|----------|
| **Short-term** | Ephemeral | `data/conversations/` | Recent 7 exchanges | `"How have you been?" → "I'm fine!"` |
| **Long-term** | Persistent | `data/profiles/` | Semantic summary, personality, goals | `"Lorenzo is a PhD student passionate about social robotics"` |

The **Memory Manager** (`memory_manager.py`) acts as a bridge between the short-term and long-term layers, performing:
- summarization,
- interest extraction,
- and incremental profile enrichment.

---

## 🔄 Data Flow

```
[ Face Detection ]
↓
[ User Identification ]
↓
[ Conversation Manager ]
↓
[ Prompt Construction ]
↓
[ LLM Response Generation ]
↓
[ Speech Output (TTS) ]
↓
[ Memory Logging → Summarization → Profile Update ]
```

---

## ⚙️ Module Responsibilities

| Module | Role | Key Functions |
|---------|------|---------------|
| `facenet_utils.py` | Face recognition & embedding comparison | `extract_embeddings()`, `compare_embeddings()` |
| `recognize_live.py` | Main control loop; integrates video, audio, and dialogue | `handle_interaction()` |
| `dialog_manager.py` | Builds context-rich prompts and handles state logic | `build_llm_prompt()` |
| `memory_manager.py` | Manages conversation history and profile summarization | `summarize_conversation()`, `update_profile_summary()` |
| `profile_manager.py` | Creates, loads, and updates persistent user profiles | `load_profile()`, `save_profile()` |
| `speech_utils.py` | Handles speech recognition and synthesis | `transcribe_audio()`, `speak_async()` |

---

## 🧩 Integration with LLM (Ollama)

SMILE uses **Ollama** as a local LLM backend.  
Each conversational cycle builds a structured prompt with:

- **User profile context**
- **Recent conversation history**
- **Dialogue state**
- **User’s current input**

The LLM then generates a coherent, personalized response that reflects the agent’s memory of previous sessions.

---

## 🧱 Data Structures

### **User Profile (JSON)**
```json
{
  "name": "Lorenzo",
  "known_since": "2025-11-02",
  "age": "25-35",
  "gender": "male",
  "occupation": "PhD student in AI and robotics",
  "interests": ["AI", "social robotics", "Cuban music"],
  "personality": "curious, reflective",
  "goals": ["complete PhD", "develop social robot interfaces"],
  "notes_summary": "Lorenzo enjoys discussing music and technology.",
  "last_update": "2025-11-03T09:45:00"
}
```

---

## 🧮 Key Design Principles

- **Privacy-first**: all data stored locally, no cloud calls
- **Explainable memory**: human-readable JSON profiles
- **Incremental learning**: every interaction improves the profile
- **Multimodal grounding**: vision + speech + text coherence
- **Resilience**: gracefully handles silence and natural interruptions

---

## 🗺️ Future Directions

- **Affective computing**: detect user emotion from speech and facial cues
- **Multi-user memory**: shared context between different recognized individuals
- **Cross-session reasoning**: merge semantic knowledge across profiles
- **Conversational grounding**: adaptive recall of past discussions

---

## 🧑‍💻 Author

**Lorenzo D'Errico**  
PhD student in AI @ Federico II  
Email: [lorenzo.derrico@unina.it]  
LinkedIn: [linkedin/lo_de06]