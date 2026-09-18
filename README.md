# Automated Rewrite Workflow for Video Content

A local-first pipeline that transforms a YouTube video into **original shorts** (9:16) with LLM-rewritten narration, programmatically regenerated data charts, and source citation — without reproducing any copyrighted footage or audio from the original source.

Built as a response to the Practical Assessment: API chaining, automated workflow, and strict cost-control design.

## Workflow

```text
YouTube URL
  → 1. Audio          yt-dlp + FFmpeg (mp3 mono 64 kbps)            cost $0
  → 2. Transcription  YouTube captions OR Groq Whisper (free, ~200x
                      faster) OR local faster-whisper as fallback   ~$0.0001
  → 3. Pre-filter     heuristic window packing 15–55s               cost $0
  → 4. Rewrite        Gemini free tier → original scripts + citation
                      "According to <source>..." + chart data (JSON)  ~$0.0001
  → 5. Charts         matplotlib regenerates each chart from data   cost $0
  → 6. Speech         edge-tts (keyless, high-quality, free)         cost $0
  → 7. Video          FFmpeg composes PNG cards + chart + voice     cost $0
  → 8. Index          Primary Pinecone + local Chroma (backup/mirror)
                      deduplication by ID and textual similarity    cost $0 (local / Pinecone free tier)
```

## Why These Choices? (Summary)

| Step | Choice | Reason |
| --- | --- | --- |
| **Audio** | Portable FFmpeg (`imageio-ffmpeg`) | No need to install system-level binaries |
| **Transcription** | YouTube Captions FIRST; then Groq Whisper `whisper-large-v3-turbo` (free tier); local whisper as fallback | Captions are free; Groq is free and ~200× faster than local CPU whisper |
| **Rewrite** | Gemini `gemini-2.5-flash` (free tier with fallbacks) | Multimodal, extremely cheap/free, highly structured JSON output |
| **Charts** | `matplotlib` | Programmatically regenerates from structured data — never takes screenshots |
| **Speech** | `edge-tts` | Free, keyless, high quality |
| **Video** | FFmpeg (card-composition + audio muxing) | No original video footage used — ensures 100% original content |
| **Index** | Pinecone (server-side embeddings `multilingual-e5-large`) + local Chroma/ONNX mirror | Deduplication by ID and textual similarity (semantic recall + n-gram Jaccard); local backup works offline or if Pinecone is unreachable |

Detailed breakdown and cost estimates: **[COSTS.md](COSTS.md)**.

## Requirements

- Python 3.11–3.14
- Internet access (YouTube download, Gemini, edge-tts)
- A **Gemini free tier** API key (https://aistudio.google.com/)

## Installation

### Windows (PowerShell)

```powershell
cd Desktop\automated_rewrite
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Linux / WSL

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` automatically installs dependencies including `local`, `index`, `gemini`, and `dev` extras.

## Configuration

```powershell
Copy-Item .env.example .env
# edit .env and insert your GEMINI_API_KEY
```

Key environment variables (all optional except the key):

| Variable | Default | Function |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | Required for rewrite + guardrail (free tier) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Rewriter & Evaluator model |
| `REELS_SOURCE_NAME` | `HEALTH 2.0` | Source name used in citations |
| `REELS_OUTPUT_LANG` | `繁體中文` | Language for script and narration |
| `REELS_TTS_VOICE` | `zh-TW-HsiaoYuNeural` | edge-tts voice |
| `REELS_EMBEDDING` | `local` | `local` (ONNX, $0) or `gemini` (Gemini embeddings API) |
| `WHISPER_MODEL` | `base` | Local whisper model (offline fallback) |
| `REELS_TRANSCRIBER` | `auto` | `auto`/`groq` → uses Groq when API key is present; `local` forces local whisper |
| `GROQ_API_KEY` | — | Groq free tier key for high-speed transcription |
| `REELS_VECTORDB` | `pinecone` | `pinecone` (primary) or `chroma` (local only) |
| `REELS_VECTORDB_BACKUP` | `chroma` | Local backup mirror; `none` disables |
| `PINECONE_API_KEY` | — | Pinecone API key (server-side `multilingual-e5-large` embeddings) |
| `REELS_DEDUPE_THRESHOLD` | `0.7` | Minimum n-gram Jaccard overlap to consider content duplicated; `0` disables |

## Usage

```bash
python main.py process "https://www.youtube.com/watch?v=VIDEO_ID" --max-clips 3
```

- `--force` reprocesses a video even if already indexed.
- `--force-transcribe` ignores cached subtitles and forces local whisper execution.
- `--max-clips N` limits the number of generated shorts; selection is managed by Gemini (the local pre-filter groups coherent transcript windows).

Search processed clips (both source and rewritten segments):

```bash
python main.py search "宽限期過後月付金額" --limit 6
```

Clear the database and index:

```bash
python main.py reset-index
```

Outputs stored in `output/videos/<video_id>/`:

```text
original.<ext>      source audio (reference only; never used in final short)
audio.mp3           normalized audio (mono 64 kbps)
analysis.json       transcript + rewritten scripts (JSON)
clips/
  01-<title>.mp4    9:16 short (cards + TTS voiceover + regenerated chart)
  narration-XX.mp3  generated voiceover
  chart-XX.png      regenerated data chart
```

## Content De-duplication

Beyond checking the YouTube ID, the pipeline **analyzes the new transcript** and compares it to already indexed ones:

1. **Semantic Recall**: The new transcript windows are embedded and searched against the vector index (Pinecone or Chroma).
2. **Exact Confirmation**: Candidates are verified using **n-gram Jaccard similarity** of normalized full texts. "Same topic" is distinguished from "same video".

If a video overlaps ≥ `REELS_DEDUPE_THRESHOLD` with an indexed transcript, processing halts (`DuplicateContent`) to prevent incurring API costs or rendering times. `--force` overrides this check.

## Copyright Compliance Policy

- **Mandatory Rewrite**: Sentence structure and wording are completely rewritten. Original narration is never copied verbatim.
- **Source Citation**: Each script begins with a clear credit ("According to HEALTH 2.0 reporting...").
- **Regenerated Charts**: Charts are programmatically plotted from scratch (using `matplotlib`) from data tables. No screenshots are taken.
- **Zero Footage Muxing**: The generated video does not reuse the original source's video or audio streams, producing 100% original creative content.

## Quality Assurance & Verification

```bash
.venv/bin/python -m pytest -q        # unit tests
.venv/bin/ruff check src tests main.py
```
