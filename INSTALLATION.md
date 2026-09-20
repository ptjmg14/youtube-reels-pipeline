# Installation & Quickstart Guide

This guide provides tested, step-by-step instructions to set up and run the automated video rewrite pipeline locally on Windows, macOS, or Linux.

---

## Prerequisites

- **Python**: 3.11, 3.12, 3.13, or 3.14
- **Git**
- **Internet Access** (for downloading the YouTube video, querying Gemini API, and edge-tts synthesis)
- **API Key**: A free-tier Google Gemini API key from [Google AI Studio](https://aistudio.google.com/) (takes 30 seconds, no credit card required).

> [!NOTE]
> **No manual FFmpeg installation required**: The project uses `imageio-ffmpeg` to provide a self-contained portable FFmpeg binary.
> **No paid APIs required**: Pinecone and Groq are completely optional; the pipeline works 100% offline/local with ChromaDB and local Whisper if those keys are not provided.

---

## Step-by-Step Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ptjmg14/youtube-reels-pipeline.git
cd youtube-reels-pipeline
```

### 2. Create and Activate Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv .venv
# If script execution is restricted on Windows, run:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

#### On macOS / Linux / WSL:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure `.env`
Copy the template configuration:
```bash
# On Linux/macOS:
cp .env.example .env

# On Windows (PowerShell):
Copy-Item .env.example .env
```

Open `.env` and add your Gemini free-tier key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
*(All other variables have working local defaults: zero-cost local ChromaDB, automatic YouTube channel attribution, and free edge-tts).*

---

## Running the Pipeline

### Process the Assessment Video
Run the automated pipeline on the assignment's source video:

```bash
python main.py process "https://www.youtube.com/watch?v=KjAI9r8tnOs" --max-clips 3
```

### What Happens During Execution:
1. **Deduplication Check**: Checks if the video ID or text is already indexed.
2. **Download & Audio Extraction**: Downloads YouTube captions (`.vtt`) or extracts downsampled mono MP3.
3. **Transcription**: Reads captions instantly or transcribes audio.
4. **Heuristic Pre-filter**: Partitions transcript into coherent 15–55s windows locally at $0 cost.
5. **Gemini Rewrite**: Selects up to 3 compelling moments and rewrites them into original narration + source citation + extracted chart data.
6. **Guardrail Evaluation**: Evaluates scripts and rewrites if needed.
7. **Assembly**: Generates programmatic `matplotlib` charts, synthesizes voiceover via `edge-tts`, and composes vertical 1080×1920 MP4 clips.
8. **Indexing**: Indexes transcript and rewritten shorts into vector store for future deduplication and search.

### Output Files
All generated outputs are saved to `output/videos/<video_id>/`:
```text
output/videos/KjAI9r8tnOs/
├── audio.mp3               # Normalized mono source audio
├── analysis.json           # Full transcript + rewritten scripts with citations
└── clips/
    ├── 01-<title>.mp4      # Final 1080×1920 vertical video (ready for Reels/Shorts)
    ├── 02-<title>.mp4
    ├── 03-<title>.mp4
    ├── narration-01.mp3    # High-quality neural voiceover
    └── chart-01.png        # Programmatically generated chart
```

---

## Verification & Testing

Verify that all unit tests pass:
```bash
python -m pytest -q
```
*(21 unit tests covering deduplication, vector storage, subtitle parsing, localization, evaluator prompts, and retry loop logic).*

Check linting and code style:
```bash
python -m ruff check src tests
```

---

## Useful Commands

- **Force re-processing**:
  ```bash
  python main.py process "https://www.youtube.com/watch?v=KjAI9r8tnOs" --force
  ```
- **Semantic search across indexed videos**:
  ```bash
  python main.py search "housing loan rates"
  ```
- **Reset local index & cache**:
  ```bash
  python main.py reset-index
  ```
