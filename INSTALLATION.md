# Installation Guide

## Requirements
- Python 3.11 or higher
- Internet connection (for YouTube, APIs, edge-tts)
- For optional local Whisper: FFmpeg installed (https://ffmpeg.org)

## Step-by-step

### 1. Clone or copy the project
```bash
cd ~/Desktop
git clone https://github.com/yourusername/youtube-reels-pipeline.git   # or just copy the folder
cd youtube-reels-pipeline
```

### 2. Create virtual environment
```bash
python -m venv .venv
# On Windows:
# .venv\Scripts\Activate.ps1
# On Linux/Mac:
# source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .[local,index,gemini,dev]
```

### 4. Configure API keys
```bash
cp .env.example .env
```

Edit `.env` and add:
```env
GEMINI_API_KEY=your_gemini_key_here
# GROQ_API_KEY=your_groq_key_here   # optional for faster transcription
```

### 5. Run the pipeline

```bash
python main.py process "https://www.youtube.com/watch?v=KjAI9r8tnOs" --max-clips 3
```

### 6. Search already processed videos
```bash
python main.py search "your keyword"
```

### Optional: Install via pip (global)
```bash
pip install -e .
reels --help
```

## Mac-specific notes
- Same as Linux (use `source .venv/bin/activate`)
- edge-tts works great on macOS
- No extra system dependencies needed beyond Python

## Troubleshooting
- "Video already processed" message = success (deduplication)
- Missing FFmpeg? Download from https://ffmpeg.org
- Gemini key not working? Create free key at https://aistudio.google.com

