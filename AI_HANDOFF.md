# AI Handoff — Automated YouTube-to-Reels

## Objective

Convert a YouTube video into original vertical clips (Reels/Shorts) with TTS narration, maintaining local processing and minimal costs.

Pipeline Flow:

```text
YouTube URL
  → yt-dlp bestaudio (audio-only, `ba/b`) + audio extraction (audio.mp3 mono 64kbps)
  → transcription (Groq whisper-large-v3-turbo if GROQ_API_KEY; else faster-whisper local)
  → textual deduplication (semantic recall + Jaccard confirmation) against Pinecone/Chroma
  → structural pre-filter (window packing 15–55s, no scoring) + Gemini selection (free start/end)
  → Gemini guardrail (paraphrase/citation/chart) with retry from source_text
  → TTS edge-tts (narration) + Pillow cards/matplotlib charts + FFmpeg → 9:16 clip (1080×1920)
  → indexing: primary Pinecone + local Chroma mirror
```

## Quick Installation (Windows / PowerShell)

The project uses a `.venv` within its own directory and has been validated on Python 3.14.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On WSL/Linux where `ensurepip` is missing (`python3-venv` is not installable without sudo):

```bash
python3 -m venv --without-pip .venv
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py
.venv/bin/python -m pip install -r requirements.txt
```

It is not necessary to install FFmpeg globally: `imageio-ffmpeg` provides a portable binary inside the environment.

### Verification

```bash
.venv/bin/python -m pytest -q           # 12 tests
.venv/bin/ruff check src tests main.py   # passes
.venv/bin/python -m pip check
```

### Extra Pinecone

`requirements.txt` installs `-e .[local,index,gemini,dev]`; the `pinecone` extra is not installed by default. To install Pinecone:

```bash
.venv/bin/python -m pip install -e '.[pinecone]'
```

## Execution

```bash
.venv/bin/python main.py process "https://www.youtube.com/watch?v=..." --max-clips 3
.venv/bin/python main.py search "topic or phrase to search"
.venv/bin/python main.py reset-index   # deletes Pinecone + Chroma mirror (recreated on next process)
```

## Configuration (`.env`, see `.env.example`)

| Variable | Default | Description |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | Required for rewrite + guardrail (mandatory) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model for rewrite/evaluator |
| `GROQ_API_KEY` | — | Fast transcription if present |
| `REELS_SOURCE_NAME` | auto (YouTube channel) | Manual override for the mandatory citation source; unset auto-detects from the video's channel |
| `REELS_OUTPUT_LANG` | `繁體中文` | Script/narration language |
| `REELS_TTS_VOICE` | `zh-TW-HsiaoYuNeural` | edge-tts voice |
| `REELS_VECTORDB` | `pinecone` | Primary DB (`pinecone` or `chroma`) |
| `REELS_VECTORDB_BACKUP` | `chroma` | Local mirror (`chroma` or `none`) |
| `REELS_DEDUPE_THRESHOLD` | `0.7` | Minimum Jaccard for DuplicateContent; `0` disables deduplication |
| `PINECONE_API_KEY` / `PINECONE_INDEX_NAME` | — / `reels` | Primary access (server-side embeddings `multilingual-e5-large`) |

## Structure and Responsibilities

| File | Responsibility |
| --- | --- |
| `src/youtube_reels/cli.py` | CLI `process`, `search`, `reset-index` |
| `pipeline.py` | Orchestrates flow, `DuplicateContent`, `_rewrite`, `_evaluate`, indexing |
| `downloader.py` | yt-dlp audio-only `ba/b` + `_extract_audio` to MP3 |
| `transcriber.py` | Groq (preferred) → faster-whisper local fallback |
| `analyzers/heuristic.py` | `HeuristicPrefilter`: structural window packing (no scoring) |
| `analyzers/rewriter.py` | Prompt + parsing: `RenderedScript` with `start`/`end`/`source_text` |
| `analyzers/evaluator.py` | Guardrail (paraphrase/citation/chart); `EvalError` |
| `dedupe.py` | `normalize`, `shingles`, `jaccard`, `transcript_chunks` |
| `vectordb.py` | `VectorStore`, `MirroredStore`, `open_vector_store`, `find_similar` |
| `pinecone_store.py` | `PineconeStore` with server-side Pinecone embeddings |
| `llm_utils.py` | `generate_with_fallback` — model fallback + retry (429/404/503) |
| `assembler.py` + `tts.py` | Cards/charts + edge-tts + FFmpeg 9:16 assembly |
| `models.py` | Domain models (`VideoAsset`, `TranscriptSegment`, `RenderedScript`, `SimilarVideo`, `ChartSpec`) |

## Produced Data

Each video is stored in `output/videos/<video_id>/`:

```text
original.<ext> (source audio reference only)
audio.mp3           (mono 64kbps)
analysis.json       (transcript + rewritten scripts with start/end/source_text)
clips/NN-<title>.mp4 (1080×1920, ~22–45s, TTS narration + charts)
```

`output/chroma/` is the local persistent backup (embedded index).

## Gemini: Models and Quota (2026)

- `generate_with_fallback` tries in order: `GEMINI_MODEL` → `gemini-2.5-flash` → `gemini-3.6-flash` → `gemini-flash-latest`.
- **`gemini-2.0-flash` was retired (404)** — do not add it back to the list.
- Free tier = 20 requests/day per model. When quota is exhausted (429), the pipeline jumps to the next model; 503 (high demand) is retried with backoff (3 rounds × 20s).
- `_retryable()` decides fallback for 429/404/quota; other ClientErrors (e.g., invalid key) propagate as clean `RewriteError`/`EvalError` → `ProcessError` on the CLI.

## Confirmed State

- **18-09-2026 — Actual E2E on WSL:** `KjAI9r8tnOs` (15 min, ZH) → 3 clips `1080×1920` of 22–23s with narration and charts.
- Actual timing: download 2.7s / transcription 13s (Groq) / dedupe 2.9s / rewrite Gemini 27.3s / guardrail 18.8s / TTS+assembly ~10–18s per clip / indexing 34.7s.
- Textual deduplication confirmed: 2nd execution of the same URL → `AlreadyProcessed` (fast-path ID); `find_similar` on the transcript returns a score of 1.0.
- `pytest`: 12 tests; `ruff check`: passes.
- Limitation: the free-tier quota (20/day/model) requires fallbacks or waiting for re-tests on the same day.

## Points of Attention for Future Work

- Do not test a YouTube URL or a Gemini key without explicit user request.
- ID deduplication (`is_processed`) triggers BEFORE textual deduplication; `--force` ignores both.
- Before updating ChromaDB/Pinecone, run verification commands: changes in the embeddings API may break the mirror.
- `find_similar` excludes `kind=transcript_full` records from candidate search (used only for Jaccard confirmation).
