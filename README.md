# MindGuardAI: Local Edge Assistant for Dementia Care

MindGuardAI is a streaming, extensible transcription and memory assistant application designed to support dementia care. Powered by [Qualcomm AI Hub Whisper Base En](https://aihub.qualcomm.com/compute/models/whisper_base_en?domain=Audio), ONNX, and a local knowledge graph (Neo4j + Graphiti), MindGuardAI enables real-time speech-to-text transcription, semantic memory storage, and retrieval using natural language queries—all running locally for privacy and reliability. MindGuardAI is designed to function as a personal memory assistant. It captures spoken information via live audio transcription, processes this information using a local Large Language Model (LLM) to understand intent (store or retrieve), and interacts with a knowledge graph (Neo4j via Graphiti) to persist and query memories.

---

## Features

- **Live Microphone Transcription:** Real-time streaming transcription using Whisper ONNX models.
- **Knowledge Graph Integration:** Store and retrieve facts, reminders, and important information using a local Neo4j database and Graphiti.
- **Natural Language Agent:** Uses LLMs (via Ollama) to decide when to store or search information.
- **Tkinter GUI:** Simple interface for manual memory entry and starting live transcription.
- **Privacy-First:** All processing is local—no cloud required.
- **Extensible:** Designed as a base for custom local chat/memory/assistant workflows.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Model Download](#model-download)
4. [Configuration](#configuration)
5. [Running the Application](#running-the-application)
6. [Usage](#usage)

---

## Requirements

- **Python 3.11.9** 
- **Windows 11 (Snapdragon X Elite)** 
- **Memory (32 GB)**
- **Neo4j** (local instance, e.g. via [Neo4j Desktop](https://neo4j.com/download/) or via docker)
- **FFmpeg** (for audio processing)
- **Ollama** (for local LLM inference)
- **ONNX Runtime** (with QNNExecutionProvider for Snapdragon X Elite, or CPU fallback)
- **Other dependencies:** See `requirements.txt`

---

## Setup

### 1. Install FFmpeg

- Download [FFmpeg for Windows](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip).
- Extract to `C:\Program Files\ffmpeg`.
- Add `C:\Program Files\ffmpeg\bin` to your system `PATH`.
- Verify by running `ffmpeg` in a new terminal.

### 2. Clone the Repository

```sh
git clone https://github.com/srinjoydutta03/MindGuardAI.git
cd MindGuardAI
```

### 3. Activate your virtual environment
```sh
python -m venv whisper-venv
# Activate (Windows)
.\whisper-venv\Scripts\activate
# Activate (Linux/Mac)
# source whisper-venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install and Start Neo4j
- Download and install Neo4j Desktop.
- Create local database using your own set credentials
- Start the database and ensure it's running at neo4j://localhost:7687.

### 5. Install and Start Ollama
- Download and install Ollama.
- Start the Ollama server.
- Pull a compatible llm and embedding model (e.g., ollama pull qwen2.5:1.5b and ollama pull nomic-embed-text).

## Model Download

Generate API Token from Qualcomm AI hub
```sh
pip install qai-hub
qai-hub configure --api_token API_TOKEN
python -m qai_hub_models.models.whisper_base_en.export --target-runtime onnx --device "Snapdragon X Elite CRD"
```
Copy the exported model files from `build` to a new directory `models` under the project root.

## Configuration

Create a config.yaml file in the project root with the following content (edit as needed):

```sh
# audio settings
"sample_rate": 16000          # Audio sample rate in Hz
"chunk_duration": 6           # Duration of each audio chunk in seconds
"channels": 1                 # Number of audio channels (1 for mono)

# processing settings
"max_workers": 4              # Number of parallel transcription workers
"silence_threshold": 0.006    # Threshold for silence detection
"queue_timeout": 1.0          # Timeout for audio queue operations

# model paths
"encoder_path": "models/WhisperEncoder.onnx"
"decoder_path": "models/WhisperDecoder.onnx"
```

## Running the application

1. Start Neo4j and Ollama
Ensure both Neo4j and Ollama servers are running.
2. Run the Application
```sh
python src/main.py
```
This will launch a Tkinter GUI for manual memory entry and a button to start live transcription.

## Usage

- Manual Memory Entry: Enter a memory and description in the GUI and click "Remember Memory" to store it in the knowledge graph.
- Live Transcription: Click "Start Listening" to begin microphone transcription. Spoken facts/questions are transcribed and processed by the agent, which decides to store or search information.
- Voice Feedback: When a relevant memory is found, the application will speak the answer aloud.

## References

This project is built on top of [Simple-Whisper-Transcription](https://github.com/thatrandomfrenchdude/simple-whisper-transcription). Many thanks to Nick for the head start.
