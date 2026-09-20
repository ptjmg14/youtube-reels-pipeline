# Architectural Decisions and Design Rationale

This document explains the technical architecture, tool choices, and cost-control strategies implemented for the **Automated Rewrite Workflow for Video Content**, specifically addressing the key evaluation criteria.

---

## 1. What Matters Most: Cost Awareness & Resource Optimization

The primary engineering focus of this pipeline is to minimize token usage, latency, and unnecessary compute while maintaining high output quality.

### Q1: For transcription, do you preprocess a long video to cut costs, or just feed the whole thing in as-is?
- **Zero-Cost First Pass (YouTube Captions)**: Before extracting or transcribing audio, the downloader probes for existing human or auto-generated `.vtt` subtitles. If found, transcription cost is **$0.00** and execution time is near-instant (<1s).
- **Audio Preprocessing & Downsampling**: When transcription is necessary, we do not feed raw video or uncompressed audio. FFmpeg extracts a **mono 64 kbps MP3** stream. This reduces file size by ~90% compared to typical video audio, drastically reducing upload bandwidth, latency, and token/file payload limits.
- **Fast-tier vs. Local Fallback**: We utilize Groq's hosted `whisper-large-v3-turbo` on its free tier (~200x faster than local CPU Whisper, $0 API cost), seamlessly falling back to offline `faster-whisper` (int8 quantized, local CPU) if offline or unkeyed.

### Q2: Script-breakdown and video generation are usually the most expensive steps — do you filter or make a judgment call before sending things through?
- **Scored Sliding-Window Pre-filter (`HeuristicPrefilter`)**: We do **not** feed a raw transcript dump into the LLM. Instead, a local Python heuristic (pure stdlib `re`, **$0 cost**) runs three passes before any API call:
  1. **Overlapping window generation** — a 50 % step sliding window replaces the old greedy non-overlapping packer, so content at window boundaries is never silently dropped.
  2. **Information-density scoring** — each window is scored on five regex signals: numeric density (numbers, percentages, CJK numerals), contrast markers (*but / 但是 / porém*), question hooks (*how / 為什麼 / como*), surprise/emotion markers (*暴增 / shocking / !*), and capitalised named-entity proxies. Higher scores indicate more quotable, engaging moments.
  3. **Top-N ranking with chronological restore** — only the top 30 windows (configurable) by score are forwarded to Gemini, keeping the rewrite prompt compact. The filtered set is then re-sorted chronologically so the narrative context reads naturally.
- **Selective Moment Picking**: The Gemini Rewriter is presented with numbered window candidates in a single prompt and selects only the top `max_clips` (default 3) most engaging moments, generating concise scripts only for those selected moments.
- **Multi-Turn Guardrail Validation**: Before generating audio or assembling video, an `Evaluator` verifies copyright compliance, paraphrase uniqueness, and factual chart grounding. If an issue is found, a targeted revision loop refines the script. This prevents wasting downstream resources (TTS synthesis and video composition) on flawed scripts.
- **Zero-Cost Programmatic Video Assembly**: Rather than using expensive cloud generative video APIs (e.g. Runway, Sora, HeyGen at $0.20–$2.00 per minute), the pipeline programmatically renders high-resolution 1080×1920 vertical video using Pillow (text cards), Matplotlib (charts), and FFmpeg. This delivers 100% original, copyright-clean video at **$0 rendering cost**.

### Q3: Do you have anything in place to stop the same video from being processed twice and racking up unnecessary cost?
Yes, a **two-tier deduplication engine**:
1. **Tier 1 — YouTube Video ID Cache (Fast Path)**: The video ID is probed before downloading media. If the vector index (Pinecone or Chroma) already contains this ID, processing halts immediately (`AlreadyProcessed`), taking <1 second and costing **$0**.
2. **Tier 2 — Semantic & N-Gram Transcript Deduplication (Content Path)**: If a video is re-uploaded by another channel or under a different URL, an ID check alone would fail. After transcription, the pipeline performs:
   - **Semantic vector recall** against indexed segments to identify candidate matches.
   - **N-gram Jaccard similarity confirmation** on normalized text.
   If similarity exceeds `REELS_DEDUPE_THRESHOLD` (default 0.7), execution halts with `DuplicateContent`, preventing redundant LLM rewriting and video rendering.

---

## 2. Mandatory Copyright & Legal Compliance

The source material is protected news/media content. The pipeline strictly complies with copyright and journalistic requirements:
1. **Verbatim Prevention & Paraphrase**: The prompt enforces complete rewording and restructuring of facts. The Evaluator guardrail checks that sentence structures differ substantially while preserving factual entities, dates, and numbers.
2. **Source Attribution**: Every clip begins with an explicit verbal and visual citation (e.g., *"According to [Channel]'s reporting..."* or *"Segundo apurado por [Canal]..."*), dynamically resolved from the video channel metadata.
3. **Programmatic Chart Regeneration**: When numerical or statistical data is present, data points are extracted into structured JSON and plotted from scratch using `matplotlib`. Screenshots or clips of original footage are never used.
4. **Zero Footage Muxing**: No original video frames or audio streams are included in the output shorts.
5. **Future Guardrail Optimization with Jev (Typesafe AI)**: A future possibility for a cheaper, reliable guardrail is investigating Jev to enforce rigid, programmatic assertions. The core architectural decision will be determining the threshold of trust: identifying which journalistic constraints require the semantic understanding of an LLM evaluator, and which can be safely offloaded to Jev's strict, zero-shot structural rules.

---


