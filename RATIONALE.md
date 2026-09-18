# RATIONALE.md

## Why use a Hybrid Deduplication Mechanism?

The pipeline employs a two-tier deduplication strategy to ensure maximum efficiency and content integrity:

1.  **YouTube Video ID Check (Primary):** This is the fastest check. If the specific ID has already been indexed, the pipeline skips processing immediately to save costs and time.
2.  **Semantic Content Similarity (Secondary):** Even if the Video ID is different (e.g., re-uploads, different channels sharing the same content), the pipeline performs a **semantic recall** followed by an **n-gram Jaccard similarity** check. This prevents processing the same core information multiple times, even when presented via different URLs.

**Why this approach was chosen:**
- **Stability:** Video ID provides a permanent anchor for specific uploads.
- **Robustness:** Semantic similarity (using shingles and embeddings) catches "re-skinned" content that a simple ID check would miss.
- **Cost Efficiency:** By downloading the transcript first, we can perform a high-fidelity check before the most expensive steps (Gemini Rewriting and Video Assembly).

## Architecture: Hybrid Vector Storage

The project implements a **Mirrored Vector Store** architecture:
- **Pinecone (Cloud):** Acts as the primary global index for distributed access.
- **ChromaDB (Local):** Acts as a high-performance local cache and backup. 
- If the cloud provider is unreachable, the system automatically falls back to the local mirror, ensuring that deduplication and search features remain "always-on."

## Language & Global Support

- **Codebase:** All core logic, documentation, and configuration are in English to maintain professional standards and compatibility.
- **Output:** The pipeline supports multi-language output (currently optimized for Traditional Chinese and English), adapting to the source material's context.
- **Embeddings:** Uses local ONNX-based models (`all-MiniLM-L6-v2`) by default, providing a free, fast, and privacy-focused way to handle vectorizations without external API calls.

## Reliability: Automated Evaluation Guardrails

Unlike simple scripts, this pipeline includes a **Guardrail Evaluation** phase:
- After Gemini generates a script, an **Evaluator** (also powered by LLM) checks for quality, structure, and adherence to constraints.
- If a script fails evaluation, the system automatically attempts a **surgical rewrite**, significantly reducing "hallucinations" or malformed JSON outputs in the final video.

## Tool Choices (Rationale)

- **yt-dlp + FFmpeg**: Industry standard for reliable media extraction.
- **Hybrid Transcription**: Prioritizes free VTT captions when available, falling back to **Groq Whisper** (for speed) or **local faster-whisper** (for privacy/offline).
- **Gemini Flash**: Chosen for its high context window and superior reasoning in the "free tier" category.
- **edge-tts**: High-quality neural voices without the cost of paid APIs like ElevenLabs.
- **Matplotlib**: Used to programmatically regenerate charts from raw data, ensuring visual consistency and preventing "blurry screenshot" issues.

## Compliance & Standards

- **Strict Attribution:** Every generated clip includes a mandatory citation of the source material.
- **Non-Derivative Rewriting:** The LLM is prompted to restructure and reword content to ensure the final product is a "new work" rather than a simple copy.
- **Ephemeral Storage:** All intermediate artifacts (audio chunks, frames) are managed within a structured output directory for easy cleanup.

