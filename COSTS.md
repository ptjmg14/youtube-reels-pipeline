# Costs and Architectural Decisions

This document details the cost-control design and flexibility choices made for this automated video pipeline, in accordance with the project requirements.

## Core Design Philosophy: Dual-Mode Architecture
The system supports two operating modes for major components, configurable via environment variables (`.env`). This allows you to choose between **Cost-Zero (Local)** and **High-Performance (API)** based on your specific volume and latency needs. Modular architecture allows for hybrid usage if we require some parts to be faster or if deemed "more reliable".

### 1. Transcription Options
- **API (Groq)**: Uses `whisper-large-v3-turbo` via Groq API. Extremely fast, near-zero latency, highly scalable.
- **Local (Whisper)**: Uses `faster-whisper` (CPU-optimized) locally. Zero cost, privacy-focused, works offline.
- *Default: `auto` (Attempts API, falls back to local if API key missing or fails).*

### 2. Vector Database Options
- **API (Pinecone)**: Serverless, cloud-native vector database. Ideal for production and semantic search across huge datasets.
- **Local (ChromaDB)**: Persistent local file-based database. Zero infrastructure cost, fully offline.
- *Default: Configurable via `REELS_VECTORDB` (`chroma` or `pinecone`).*

## Cost Awareness Implementation
- **De-duplication**: two checks. (1) By YouTube ID before any download (cheap fast-path). (2) By **textual similarity of the transcript** after transcription and **before** Gemini rewriting/rendering: window embeddings recall candidates (Pinecone primary, local Chroma mirror as fallback) and n-gram Jaccard confirms a real duplicate. A match skips the API and render steps entirely.
- **Batching**: Embedding generation uses batching to comply with API limits (`_EMBEDDING_BATCH=96`), reducing requests and preventing timeouts.
- **Guardrails**: The `Evaluator` module runs after AI script generation to verify quality, copyright compliance, and chart veracity before proceeding to expensive assembly (TTS + FFmpeg). *Note: This adds a secondary, highly efficient LLM call that only processes the pre-filtered scripts (not the full transcript), ensuring quality with minimal extra token overhead.*

## Component Reasoning
| Step | Tool/Service | Cost-Zero Option | API Performance Option |
| :--- | :--- | :--- | :--- |
| **Download** | `yt-dlp` | `yt-dlp` (Local) | `yt-dlp` (Local) |
| **Transcription**| Whisper | `faster-whisper` | Groq API |
| **Rewriter** | Gemini | Gemini API (Free Tier) | Gemini API (Paid/Pro) |
| **Database** | Chroma/Pinecone | ChromaDB | Pinecone |
| **Assembly** | FFmpeg | FFmpeg | FFmpeg |

## How to Run Locally
1. Install system-level FFmpeg.
2. Install Python dependencies: `pip install -e .` (plus `.[local]` or `.[pinecone]` as needed).
3. Copy `.env.example` to `.env` and configure your preferences:
   - `REELS_TRANSCRIBER=auto` (or `groq`)
   - `REELS_VECTORDB=chroma` (or `pinecone`)
4. Run the pipeline: `python main.py process <youtube_url>`.
