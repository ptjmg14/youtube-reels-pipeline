# Pipeline — Automated Rewrite Workflow

Actual architecture and execution flow diagram (reflects code in `src/youtube_reels/`).

```mermaid
flowchart TD
    A[YouTube URL] --> B[probe_video_id<br/>yt-dlp, no download]

    B -->|already indexed and no --force| C[AlreadyProcessed<br/>The video is already indexed]
    B -->|new / --force| DG[download]

    subgraph DG [Download · $0 · cached artifacts]
        D1{original.* in cache?}
        D1 -->|no| D2[yt-dlp: bestaudio<br/>+ captions pt/en/zh VTT]
        D1 -->|yes, no captions| D3[yt-dlp captions only<br/>skip_download]
        D1 -->|yes, with captions| D4[reuse artifacts]
        D2 --> D4
        D3 --> D4
        D4 --> D5[FFmpeg extracts audio<br/>audio.mp3 mono 64 kbps]
    end

    D5 --> E{VTT caption available?}

    E -->|yes| F[parse_vtt → segments<br/>free and exact]
    E -->|no| GT[transcribe]

    subgraph GT [Transcription · ~10s · free tier]
        G1{REELS_TRANSCRIBER auto/groq<br/>and GROQ_API_KEY?}
        G1 -->|yes| G2[Groq whisper-large-v3-turbo<br/>POST /audio/transcriptions<br/>verbose_json → segments]
        G2 -->|failure| G3[faster-whisper local<br/>int8 + VAD + beam=1]
        G1 -->|local / no key| G3
    end

    F --> H[HeuristicPrefilter<br/>window packing 15–55s<br/>· $0]
    G2 -->|segments| H
    G3 -->|segments| H

    H --> GW[Gemini rewrite]

    subgraph GW [Selection + rewrite · ~5s · mandatory · free tier]
        W1[Prompt: pick the N best moments<br/>with free start/end (cross-window)<br/>rewrite structure and wording<br/>citation + chart data in JSON<br/>numbers anti-hallucination]
        W1 --> W2[genai → RenderedScripts<br/>narration + citation + chart_spec + start/end + source_text]
        W2 --> W3[missing citation → fallback<br/>According to &lt;channel&gt; reporting]
    end

    W3 --> J[analysis.json<br/>transcript + scripts + citations]

    W3 --> RD[TTS + assembly]

    subgraph RD [Render · ~30s · $0]
        R1[edge-tts zh-TW-HsiaoYuNeural<br/>narration-NN.mp3 · retry 3×]
        R1 --> R2[Pillow: 3 cards 9:16<br/>title / content+chart / CTA]
        R3[matplotlib regenerates chart<br/>chart-NN.png · never screenshot]
        R2 --> R4[FFmpeg: cards + audio<br/>fades + -shortest → NN-.mp4]
        R3 --> R4
    end

    R4 --> L[VectorStore.index<br/>ChromaDB: transcript + scripts<br/>local ONNX embeddings]

    L --> M[output/videos/&lt;id&gt;/clips/NN-&lt;title&gt;.mp4<br/>original shorts 9:16]
    L --> N[search main.py search<br/>SOURCE / REWRITTEN labels]

    C -.-> O[Dedupe: $0 cost on reprocessing]
```

## Phases and Cost (per ~16 min video)

| Phase | Component | Measured Time | Cost |
| --- | --- | --- | --- |
| Download | yt-dlp bestaudio + FFmpeg | 2–4s (cache) | $0 |
| Transcription | Groq `whisper-large-v3-turbo` (fallback: local whisper) | **9.6s** | $0 (free tier) |
| Pre-filter | heuristic `HeuristicPrefilter` | <0.1s | $0 |
| Rewrite | Gemini `gemini-2.5-flash` | ~5s | $0–$0.0001 |
| TTS + assembly | edge-tts + Pillow + FFmpeg | 13–85s (CDN varies) | $0 |
| Indexing | ChromaDB + ONNX | ~21s | $0 |

Copyright rules guaranteed in the **Rewrite** phase: original structure/wording are never reproduced, each script begins with a source citation, and charts are programmatically regenerated from scratch from the extracted data.

---

Equivalent rendered diagram: copy the `mermaid` block to <https://mermaid.live> or use `mmdc` (mermaid-cli) to generate SVG/PNG.
