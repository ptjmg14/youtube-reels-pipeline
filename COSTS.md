# Cost Estimation & Architectural Economics

This document provides a comprehensive cost model, numerical estimates, and engineering trade-offs for the **Automated Rewrite Workflow for Video Content**.

---

## 1. Cost Estimate Per Video (15-Minute Source Video → 3 Shorts)

Using the assignment source video (`https://www.youtube.com/watch?v=KjAI9r8tnOs`, duration ~15 minutes, generating 3 vertical clips of 20–30s):

| Pipeline Stage | Selected Tool / Strategy | Cost (Current Pipeline) | Paid API Commercial Equivalent |
| :--- | :--- | :--- | :--- |
| **1. Audio Extraction** | `yt-dlp` + portable FFmpeg (mono 64k) | **$0.00** (Local CPU) | $0.00 |
| **2. Transcription** | YouTube Captions ($0) / Groq Whisper ($0 free tier) | **$0.00** (or ~$0.0001) | OpenAI Whisper API: $0.006/min × 15 = **$0.0900** |
| **3. Pre-filtering** | Heuristic window packing (Python, zero LLM) | **$0.00** (Local CPU) | LLM Pre-pass (GPT-4o): ~$0.0200 |
| **4. LLM Rewriting** | Gemini 2.5 Flash (~3.5k input tokens, ~1k output) | **$0.00** (Free tier) or **$0.0007** | Claude 3.5 Sonnet / GPT-4o: **$0.0350** |
| **5. Guardrail Evaluation** | Gemini 2.5 Flash (Targeted verification) | **$0.00** (Free tier) or **$0.0002** | Secondary LLM pass: **$0.0100** |
| **6. Chart Generation** | Programmatic `matplotlib` (zero API) | **$0.00** (Local CPU) | QuickChart API / Plotly Cloud: ~$0.0050 |
| **7. TTS Voiceover** | `edge-tts` (Microsoft Neural, keyless) | **$0.00** (Free) | ElevenLabs (~1,500 chars): **$0.0450** |
| **8. Video Assembly** | Pillow cards + FFmpeg 1080×1920 60fps | **$0.00** (Local CPU) | Cloud Video API (HeyGen/Runway): ~$1.5000 |
| **9. Dedupe & Indexing** | Local Chroma (ONNX) + Pinecone Free Tier | **$0.00** (Free tier) | Dedicated Hosted Vector DB: ~$0.0100 |
| **TOTAL PER VIDEO** | | **$0.00** (Free Tier) / **~$0.001** (Paid scale) | **~$1.71 to $3.50+** |

> [!TIP]
> **Key Takeaway**: By combining local deterministic processing (Pillow, Matplotlib, FFmpeg, ONNX), free captions, and free-tier APIs (Groq, Gemini, edge-tts), **the operational cost per video is essentially $0.00 for demo and batch scales**, and less than **$0.001** if paying standard Gemini Flash tokens. This represents a **99.9% cost reduction** compared to naive cloud video generation pipelines.

---

## 2. Cost at Scale (Projections)

| Monthly Volume | Naive Cloud Pipeline (OpenAI + ElevenLabs + HeyGen) | Our Optimized Pipeline (Free Tier / Hybrid) | Our Pipeline at Enterprise Scale (Paid APIs) |
| :--- | :--- | :--- | :--- |
| **10 videos / mo** | ~$17.10 | **$0.00** | ~$0.01 |
| **100 videos / mo** | ~$171.00 | **$0.00** (within free tier quotas) | ~$0.10 |
| **1,000 videos / mo**| ~$1,710.00 | ~$0.90 (minor Gemini API overflow) | ~$1.00 |
| **10,000 videos / mo**| ~$17,100.00 | ~$9.50 (Gemini paid tier + cloud runner) | ~$10.00 |

---

## 3. Detailed Cost-Control Mechanisms

### A. Two-Tier Deduplication
- **YouTube Video ID Cache**: Instant check against the vector store before downloading the video. Avoids 100% of download and transcription compute for previously processed videos.
- **Transcript Semantic & N-Gram Check**: Embeds transcript windows and checks Jaccard n-gram similarity against already indexed material. Catches cross-channel re-uploads and reposts before any LLM rewrite or video rendering occurs.

### B. Heuristic Pre-Filtering
- Rather than sending the full 15-minute transcript through multiple LLM passes to identify clip candidates, a local Python heuristic (`HeuristicPrefilter`) packs sentences into 15–55s candidate windows based on punctuation and timestamp continuity.
- Gemini receives the candidate blocks in a single, structured prompt and only generates output for the `max_clips` specified.

### C. Programmatic Media Generation
- **Charts**: Rendered cleanly in seconds with `matplotlib` directly from numbers extracted by the rewriter. Zero third-party image/chart API costs.
- **TTS**: `edge-tts` provides human-grade Microsoft Azure Neural voice synthesis with zero API tokens or subscriptions.
- **Composition**: FFmpeg composes vertical 1080×1920 clips with fade transitions locally on CPU in ~10 seconds per clip.

---

## 4. Operational Requirements

1. **Mandatory**: A Gemini API key on Google AI Studio's **free tier** (zero credit card required).
2. **Optional**: Groq API key for high-speed transcription (free tier, 20 requests/minute).
3. **Optional**: Pinecone API key (starter free tier with 100k vector storage). If omitted, the pipeline automatically falls back to local ChromaDB ($0 setup).

