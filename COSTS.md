# Cost Estimation & Architectural Economics

This document provides a comprehensive cost model, numerical estimates, and engineering trade-offs for the **Automated Rewrite Workflow for Video Content**.

---

## 1. Cost Estimate Per Video (15-Minute Source Video → 3 Shorts)

Using the assignment source video (`https://www.youtube.com/watch?v=KjAI9r8tnOs`, duration ~15 minutes, generating 3 vertical clips of 20–30s):

| Pipeline Stage | Selected Tool / Strategy | Cost (Current Pipeline) | Paid API Commercial Equivalent |
| :--- | :--- | :--- | :--- |
| **1. Audio Extraction** | `yt-dlp` + portable FFmpeg (mono 64k) | **$0.00** (Local CPU) | $0.00 |
| **2. Transcription** | YouTube Captions ($0) / Groq Whisper ($0 free tier) | **$0.00** (or ~$0.0001) | OpenAI Whisper API: $0.006/min × 15 = **$0.0900** |
| **3. Pre-filtering** | Scored sliding-window heuristic (pure `re`, zero LLM) | **$0.00** (Local CPU) | LLM Pre-pass (GPT-4o): ~$0.0200 |
| **4. LLM Rewriting** | Gemini 2.5 Flash (~3.5k input tokens, ~1k output) + Groq free-tier fallback | **$0.00** (Free tier) or **$0.0007** | Claude 3.5 Sonnet / GPT-4o: **$0.0350** |
| **5. Guardrail Evaluation** | Gemini 2.5 Flash (Targeted verification) | **$0.00** (Free tier) or **$0.0002** | Secondary LLM pass: **$0.0100** |
| **6. Chart Generation** | Programmatic `matplotlib` (zero API) | **$0.00** (Local CPU) | QuickChart API / Plotly Cloud: ~$0.0050 |
| **7. TTS Voiceover** | `edge-tts` (Microsoft Neural, keyless) | **$0.00** (Free) | ElevenLabs (~1,500 chars): **$0.0450** |
| **8. Video Assembly** | Pillow cards + FFmpeg 1080×1920 60fps | **$0.00** (Local CPU) | Cloud Video API (HeyGen/Runway): ~$1.5000 |
| **9. Dedupe & Indexing** | Local Chroma (ONNX) + Pinecone Free Tier | **$0.00** (Free tier) | Dedicated Hosted Vector DB: ~$0.0100 |
| **TOTAL PER VIDEO** | | **$0.00** (Free Tier) / **~$0.001** (Paid scale) | **~$1.71 to $3.50+** |

> [!TIP]
> **Key Takeaway**: By combining local deterministic processing (Pillow, Matplotlib, FFmpeg, ONNX), free captions, and free-tier APIs (Groq, Gemini, edge-tts), **the operational cost per video is essentially $0.00 for demo and batch scales**, and less than **$0.001** if paying standard Gemini Flash tokens. This represents a **99.9% cost reduction** compared to naive cloud video generation pipelines.
>
> **When would I pay?** My preference is to default to free-tier and local processing at every step — and only introduce paid APIs where the quality gap is large enough to justify it so an hybrid approach. The one area where that threshold is clearly met is **video generation itself**: if the goal is toward maximising virality, free programmatic assembly with Pillow/FFmpeg hits a ceiling. Similarly, if volume or latency requirements outgrow free-tier quotas. If over-time or after error analysis we deem LLM rewriting and guardrail evaluation to be sup-par we can also change to a better LLM without too much impact to cost.


---

## 2. Detailed Cost-Control Mechanisms

### A. Two-Tier Deduplication
- **YouTube Video ID Cache**: Instant check against the vector store before downloading the video. Avoids 100% of download and transcription compute for previously processed videos.
- **Transcript Semantic & N-Gram Check**: Embeds transcript windows and checks Jaccard n-gram similarity against already indexed material. Catches cross-channel re-uploads and reposts before any LLM rewrite or video rendering occurs.

### B. Heuristic Pre-Filtering
- Rather than dumping the raw transcript into an LLM, a local Python heuristic (`HeuristicPrefilter`) generates overlapping 15–55s windows with a **50 % step** (so no content at window boundaries is missed), then **scores** each window on five information-density signals: numeric density, contrast markers, question hooks, surprise/emotion language, and capitalised named-entity proxies — all via fast stdlib `re`, zero API cost.
- Only the **top 30 windows by score** (configurable) are forwarded to Gemini in a single structured prompt. The LLM then selects and writes only the `max_clips` specified.
- This two-stage approach (local scoring → single LLM pass) eliminates multi-pass LLM brainstorming while surfacing higher-quality candidates than a simple greedy packer.

### C. Programmatic Media Generation
- **Charts**: Rendered cleanly in seconds with `matplotlib` directly from numbers extracted by the rewriter. Zero third-party image/chart API costs.
- **TTS**: `edge-tts` provides human-grade Microsoft Azure Neural voice synthesis with zero API tokens or subscriptions.
- **Composition**: FFmpeg composes vertical 1080×1920 clips with fade transitions locally on CPU in ~5 seconds per clip (`-preset veryfast -crf 23`).

---

## 3. Operational Requirements

1. **Mandatory**: A Gemini API key on Google AI Studio's **free tier** (zero credit card required).
2. **Optional**: Groq API key for high-speed transcription (free tier, 20 requests/minute) and as the free-tier LLM fallback for rewriting/guardrail when Gemini's daily quota is exhausted.
3. **Optional**: Pinecone API key (starter free tier with 100k vector storage). If omitted, the pipeline automatically falls back to local ChromaDB ($0 setup).

