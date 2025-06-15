# MindGuardAI Technical Deep Dive

This document provides a deeper technical explanation of the core components within the MindGuardAI `src` directory.

## Core Components

The system is primarily composed of three Python modules: `main.py`, `LiveTranscriber.py`, and `model.py`.

### 1. `main.py` - Application Core & Agent Logic

This module orchestrates the overall application flow, integrating transcription, LLM-based decision making, knowledge graph operations, and user interaction.

**Key Technical Aspects:**

*   **Knowledge Graph Interaction (Graphiti & Neo4j):**
    *   Uses the `graphiti_core` library to interface with a Neo4j graph database.
    *   **Initialization:** A `Graphiti` instance is created with Neo4j connection parameters (URI, user, password), an LLM client, an embedder (OpenAI compatible, using a local Ollama endpoint for `nomic-embed-text`), and a cross-encoder/reranker.
    *   **`add_episodes(content: str, ep_description: str)`:**
        *   Asynchronously adds new "episodes" (pieces of information) to the knowledge graph.
        *   Each episode is stored with a unique name (e.g., "Episode 1"), its content, a source type (`EpisodeType.text`), a reference timestamp (`datetime.now(timezone.utc)`), and a user-provided description.
        *   A global `episode_counter` ensures unique episode naming.
    *   **`search_episodes(query: str) -> list[GraphitiSearchResult]`:**
        *   Asynchronously searches the knowledge graph using the `graphiti.search()` method.
        *   Results are formatted into `GraphitiSearchResult` Pydantic models, including the fact, UUID, and optional validity timestamps.
        *   A basic keyword-based relevance check is performed on the top search result to filter out potentially irrelevant matches before returning them. If no keywords from the query are found in the retrieved fact, the result is discarded.
*   **LLM Agent (Ollama):**
    *   Integrates with a local LLM (e.g., `qwen2.5:1.5b`) via the `ollama` Python library's `AsyncClient`.
    *   **Tool Definition (`get_ollama_tool_schemas()`):**
        *   Defines two primary tools for the LLM: `add_episodes` and `search_episodes`.
        *   These schemas describe the function's purpose, parameters (name, type, description), and required arguments, enabling the LLM to decide which tool to call and with what arguments based on user input.
    *   **Agentic Processing (`process_transcript_with_agent()`):**
        *   This asynchronous function is triggered by new transcripts.
        *   It constructs a message list with a system prompt and the user's transcript.
        *   The system prompt guides the LLM to decide whether to store information (call `add_episodes`), retrieve information (call `search_episodes`), or do nothing if the input is not actionable.
        *   It makes an API call to `ollama.AsyncClient().chat()` with the messages and tool schemas.
        *   If the LLM's response includes `tool_calls`, the corresponding local Python function (`add_episodes` or `search_episodes`) is invoked with the arguments provided by the LLM.
*   **Asynchronous Operations (`asyncio`):**
    *   The core logic heavily relies on `asyncio` for non-blocking operations, especially for LLM calls and Graphiti interactions.
    *   An event loop (`event_loop`) is managed globally and used to run asynchronous tasks from synchronous contexts (like the `handle_new_transcript` callback) via `asyncio.run_coroutine_threadsafe()`.
    *   `main_async_runner()` is the primary async function that initializes the system (Graphiti indices, LiveTranscriber) and keeps the application alive.
*   **Text-to-Speech (TTS):**
    *   Uses `pyttsx3` to vocalize the facts retrieved from the knowledge graph during a search operation. The speech rate is set to 150.
*   **Transcript Handling (`handle_new_transcript(transcript: str)`):**
    *   This function serves as a callback for the `LiveTranscriber`.
    *   When a new, non-empty transcript is received, it schedules `process_transcript_with_agent()` to run on the main event loop.
*   **GUI (Tkinter):**
    *   A simple Tkinter-based GUI allows:
        *   Manually inputting memory content and description, then calling `add_episodes`.
        *   Starting the live transcription process by initiating `main_async_runner` in a separate thread.
    *   GUI operations that trigger async functions (like `remember_memory`) attempt to use the existing asyncio event loop if available or run a new one.
*   **Shutdown Logic:**
    *   Handles `KeyboardInterrupt` for graceful shutdown.
    *   Stops the `LiveTranscriber` and joins its threads.

### 2. `LiveTranscriber.py` - Real-time Speech-to-Text

This module is responsible for capturing audio from the microphone, processing it in chunks, and transcribing it using a Whisper ONNX model.

**Key Technical Aspects:**

*   **Configuration (`config.yaml`):**
    *   Loads parameters like `sample_rate`, `chunk_duration`, `channels`, `max_workers` for transcription, `silence_threshold`, and paths to ONNX model files (`encoder_path`, `decoder_path`).
*   **Audio Input (`sounddevice`):**
    *   The `record_audio` function uses `sounddevice.InputStream` to capture audio.
    *   The `audio_callback` function is invoked by `sounddevice` with new audio data, which is then put into a thread-safe `queue.Queue` (`audio_queue`).
*   **Audio Processing (`process_audio`):**
    *   Runs in a separate thread.
    *   Continuously retrieves audio chunks from `audio_queue`.
    *   Maintains a `buffer` to accumulate audio data until it reaches `chunk_samples` (derived from `sample_rate` and `chunk_duration`).
    *   Once enough data is buffered, a `current_chunk` is extracted for transcription.
*   **Transcription Engine:**
    *   Uses `WhisperApp` from `qai_hub_models.models.whisper_base_en` which itself uses the `WhisperBaseEnONNX` class from `model.py`.
    *   The `process_transcription` function takes an audio chunk and transcribes it.
    *   **Silence Detection:** Only processes chunks where the mean absolute amplitude (`np.abs(chunk).mean()`) exceeds `silence_threshold` to avoid transcribing silence or background noise.
*   **Concurrency and Multithreading:**
    *   **Recording Thread:** `record_audio` runs in its own thread.
    *   **Processing Thread:** `process_audio` runs in its own thread.
    *   **Transcription Workers:** `ThreadPoolExecutor` with `max_workers` is used within `process_audio` to submit audio chunks to `process_transcription` for parallel transcription. This allows multiple chunks to be transcribed concurrently if the hardware supports it.
    *   **`stop_event` (threading.Event):** Used to signal all threads to terminate gracefully.
*   **Callback Mechanism (`transcript_callback`):**
    *   If a `transcript_callback` function is provided during `LiveTranscriber` initialization (as `main.py` does with `handle_new_transcript`), it's called with the transcribed text.

### 3. `model.py` - ONNX Model Wrappers for Whisper

This module provides wrappers for running Whisper encoder and decoder ONNX models using ONNX Runtime, specifically configured for Qualcomm's QNN Execution Provider.

**Key Technical Aspects:**

*   **Whisper Model Wrappers:**
    *   **`ONNXEncoderWrapper`:**
        *   Initializes an ONNX session for the Whisper encoder model using `get_onnxruntime_session_with_qnn_ep`.
        *   The `__call__` method executes the encoder model (`self.session.run`) with the input audio.
    *   **`ONNXDecoderWrapper`:**
        *   Initializes an ONNX session for the Whisper decoder model.
        *   The `__call__` method executes the decoder model, taking various inputs like `x` (token IDs), `index`, and cache states (`k_cache_cross`, `v_cache_cross`, etc.).
*   **`WhisperBaseEnONNX(Whisper)` Class:**
    *   Inherits from `qai_hub_models.models._shared.whisper.model.Whisper`.
    *   Initializes the base `Whisper` class with instances of `ONNXEncoderWrapper` and `ONNXDecoderWrapper`.
    *   Specifies architectural parameters of the Whisper Base English model (e.g., `num_decoder_blocks=6`, `num_heads=8`, `attention_dim=512`). This class effectively bridges the generic Whisper logic from `qai_hub_models` with the ONNX-based, QNN-accelerated execution.

## Workflow Summary

1.  **Audio Capture (`LiveTranscriber`):** Microphone audio is continuously captured, chunked, and queued.
2.  **Transcription (`LiveTranscriber` & `model.py`):** Audio chunks are dequeued and transcribed in parallel worker threads using the Whisper ONNX models (via `WhisperBaseEnONNX` running on QNN EP).
3.  **Transcript Handling (`main.py`):** Transcribed text is passed to `handle_new_transcript`.
4.  **Agent Processing (`main.py`):**
    *   `process_transcript_with_agent` sends the transcript to the Ollama LLM.
    *   The LLM decides whether to call `add_episodes` (to store information) or `search_episodes` (to retrieve information) based on its understanding of the transcript and the provided tool schemas.
5.  **Knowledge Graph Operation (`main.py` & `graphiti_core`):**
    *   If `add_episodes` is called, the information is added to the Neo4j graph.
    *   If `search_episodes` is called, a query is performed against the graph.
6.  **Response/Action (`main.py`):**
    *   For `add_episodes`, a confirmation is printed.
    *   For `search_episodes`, if relevant results are found, the primary fact is spoken aloud using `pyttsx3`.
7.  **Manual Input (Tkinter in `main.py`):** Users can also manually input memories or start/stop the listening process via the GUI.

## Configuration

*   **`config.yaml` (used by `LiveTranscriber.py`):** Contains settings for audio processing (sample rate, chunk duration), transcription (worker count, silence threshold), and paths to the Whisper ONNX encoder/decoder models.
*   **Environment Variables / Hardcoded Values (in `main.py`):** Neo4j connection details (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`) and LLM/Embedder configurations (API keys, model names, base URLs for Ollama) are set directly in `main.py` or expected as environment variables.

## Key Dependencies

*   `asyncio`: For asynchronous programming.
*   `threading`, `queue`: For multithreaded audio processing.
*   `sounddevice`: For audio input.
*   `numpy`: For numerical operations, especially audio data manipulation.
*   `pyyaml`: For loading `config.yaml`.
*   `ollama`: Client for interacting with Ollama LLMs.
*   `graphiti_core`: Library for interacting with the Neo4j knowledge graph.
*   `pydantic`: For data validation and settings management (used by Graphiti and for defining data structures).
*   `onnxruntime`: For running ONNX models.
*   `qai_hub_models`: Contains the Whisper model application logic that `model.py` builds upon. (Assumed, as `WhisperApp` and `Whisper` base class are used).
*   `pyttsx3`: For text-to-speech output.
*   `tkinter`: For the optional GUI.