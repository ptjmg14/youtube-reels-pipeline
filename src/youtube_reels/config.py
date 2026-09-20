from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Environment-driven configuration (all values overridable via .env)."""

    data_dir: Path
    gemini_api_key: str | None
    gemini_model: str
    gemini_embedding_model: str
    embedding_provider: str
    whisper_model: str
    transcriber: str
    groq_api_key: str | None
    groq_model: str
    llm_fallback: str
    groq_rewrite_models: tuple[str, ...]
    vectordb_backend: str
    vectordb_backup: str
    dedupe_threshold: float
    pinecone_api_key: str | None
    pinecone_index_name: str
    pinecone_region: str
    pinecone_embedding_model: str
    source_name: str
    output_language: str
    tts_voice: str
    tts_rate: str

    @classmethod
    def from_env(cls) -> Settings:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        return cls(
            data_dir=Path(os.getenv("REELS_DATA_DIR", "output")).resolve(),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            gemini_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004"),
            embedding_provider=os.getenv("REELS_EMBEDDING", "local").lower(),
            whisper_model=os.getenv("WHISPER_MODEL", "base"),
            transcriber=os.getenv("REELS_TRANSCRIBER", "auto").lower(),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_model=os.getenv("GROQ_MODEL", "whisper-large-v3-turbo"),
            llm_fallback=os.getenv("REELS_LLM_FALLBACK", "groq").lower(),
            groq_rewrite_models=_split_models(
                os.getenv(
                    "REELS_LLM_FALLBACK_MODELS",
                    "llama-3.3-70b-versatile, openai/gpt-oss-120b, llama-3.1-8b-instant",
                )
            ),
            vectordb_backend=os.getenv("REELS_VECTORDB", "pinecone").lower(),
            vectordb_backup=os.getenv("REELS_VECTORDB_BACKUP", "chroma").lower(),
            dedupe_threshold=float(os.getenv("REELS_DEDUPE_THRESHOLD", "0.7")),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "reels"),
            pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
            pinecone_embedding_model=os.getenv(
                "PINECONE_EMBEDDING_MODEL", "multilingual-e5-large"
            ),
            source_name=os.getenv("REELS_SOURCE_NAME", "").strip(),
            output_language=os.getenv("REELS_OUTPUT_LANG", "繁體中文"),
            tts_voice=os.getenv("REELS_TTS_VOICE", "zh-TW-HsiaoYuNeural"),
            tts_rate=os.getenv("REELS_TTS_RATE", "+5%"),
        )

    def video_dir(self, video_id: str) -> Path:
        path = self.data_dir / "videos" / video_id
        path.mkdir(parents=True, exist_ok=True)
        return path


def _split_models(raw: str) -> tuple[str, ...]:
    return tuple(
        model.strip()
        for model in raw.split(",")
        if model.strip()
    )
