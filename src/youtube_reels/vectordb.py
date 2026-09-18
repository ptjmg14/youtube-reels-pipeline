from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .config import Settings
from .dedupe import jaccard, shingles, transcript_chunks, transcript_text
from .models import RenderedScript, SimilarVideo, TranscriptSegment, VideoAsset

if TYPE_CHECKING:
    from .pinecone_store import PineconeStore


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        try:
            import chromadb
        except ImportError as error:  # pragma: no cover - dependency guard
            raise RuntimeError("Install the local index: pip install -e '.[index]'") from error
        # Chroma's built-in ONNX embedding keeps the local path free from Torch.
        # Its default cache is in the user profile, which may be policy-protected.
        embedding_function: Any = ProjectOnnxEmbeddingFunction(
            settings.data_dir / "embedding-models"
        )
        if settings.embedding_provider == "gemini":
            if not settings.gemini_api_key:
                raise RuntimeError("REELS_EMBEDDING=gemini requires GEMINI_API_KEY.")
            embedding_function = GeminiEmbeddingFunction(
                settings.gemini_api_key, settings.gemini_embedding_model
            )
        elif settings.embedding_provider != "local":
            raise RuntimeError(
                f"Unknown embedding provider: {settings.embedding_provider}"
            )
        self._collection_name = "video_segments_v2"
        self._client = chromadb.PersistentClient(path=str(settings.data_dir / "chroma"))
        self._embedding_function = embedding_function
        self.collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=embedding_function,
        )

    def is_processed(self, video_id: str) -> bool:
        return bool(self.collection.get(ids=[f"{video_id}:transcript:0"], include=[])["ids"])

    def reset(self) -> None:
        """Drop the whole collection (recreated automatically on next use)."""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception as error:  # noqa: BLE001 - collection may not exist yet
            print(f"Local collection already absent ({error}).")
        self.collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_function,
        )

    def index(
        self,
        asset: VideoAsset,
        transcript: list[TranscriptSegment],
        scripts: list[RenderedScript],
    ) -> None:
        documents: list[str] = []
        ids: list[str] = []
        metadata: list[dict[str, Any]] = []
        for index, segment in enumerate(transcript):
            documents.append(segment.text)
            ids.append(f"{asset.video_id}:transcript:{index}")
            metadata.append(
                {
                    "video_id": asset.video_id,
                    "title": asset.title,
                    "kind": "transcript",
                    "start": segment.start,
                    "end": segment.end,
                }
            )
        for script in scripts:
            documents.append(script.narration)
            ids.append(f"{asset.video_id}:script:{script.number}")
            metadata.append(
                {
                    "video_id": asset.video_id,
                    "title": asset.title,
                    "kind": "script",
                    "hook": script.hook,
                    "clip": script.number,
                }
            )
        full_text = transcript_text(transcript)
        if full_text:
            documents.append(full_text)
            ids.append(f"{asset.video_id}:transcript:full")
            metadata.append(
                {
                    "video_id": asset.video_id,
                    "title": asset.title,
                    "kind": "transcript_full",
                }
            )
        if not documents:
            return
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadata)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where={"kind": {"$ne": "transcript_full"}},
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"text": document, "metadata": metadata, "distance": distance}
            for document, metadata, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def find_similar(
        self, transcript: list[TranscriptSegment], threshold: float, top_k: int = 3
    ) -> SimilarVideo | None:
        """Recall candidate videos semantically, then confirm exact overlap.

        Embedding search alone cannot separate "same topic" from "same video";
        the returned candidates are therefore verified with character n-gram
        Jaccard over the stored full transcript.
        """
        chunks = transcript_chunks(transcript)
        if not chunks:
            return None
        result = self.collection.query(
            query_texts=chunks, n_results=top_k, include=["metadatas"]
        )
        candidate_ids: list[str] = []
        for metadatas in result["metadatas"] or []:
            for metadata in metadatas or []:
                video_id = metadata.get("video_id")
                if video_id and video_id not in candidate_ids:
                    candidate_ids.append(video_id)
        query_shingles = shingles(" ".join(chunks))
        best: SimilarVideo | None = None
        for video_id in candidate_ids:
            record = self.collection.get(
                ids=[f"{video_id}:transcript:full"], include=["documents", "metadatas"]
            )
            documents = record.get("documents") or []
            if not documents or not documents[0]:
                continue
            score = jaccard(query_shingles, shingles(documents[0]))
            if best is None or score > best.score:
                title = (record.get("metadatas") or [{}])[0].get("title", "")
                best = SimilarVideo(video_id=video_id, title=title, score=score)
        return best if best and best.score >= threshold else None


class GeminiEmbeddingFunction:
    """Chroma adapter that uses Gemini embeddings only when explicitly selected."""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        try:
            from google import genai
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("Install Gemini embeddings: pip install -e '.[gemini]'") from error
        client = genai.Client(api_key=self.api_key)
        response = client.models.embed_content(model=self.model, contents=list(input))
        return [list(item.values) for item in response.embeddings]


class ProjectOnnxEmbeddingFunction:
    """Chroma's default embedding model with a project-writable cache."""

    def __new__(cls, cache_root: Any) -> Any:
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

        class _ProjectOnnx(ONNXMiniLM_L6_V2):
            def name(self) -> str:
                # Retain Chroma's default identifier for existing collections.
                return "default"

        embedding = _ProjectOnnx()
        embedding.DOWNLOAD_PATH = cache_root / embedding.MODEL_NAME
        return embedding


class MirroredStore:
    """Hosted primary index with a local Chroma mirror used as backup.

    Both stores share the same id scheme (``{video_id}:transcript:N`` and
    ``{video_id}:script:N``), so the local Chroma collection is a complete
    copy. Reads (``is_processed``/``search``) hit the primary and fall back to
    the mirror when the hosted index is unreachable; writes go to both so the
    backup stays current even if the primary later goes down.
    """

    def __init__(self, primary: Any, backup: VectorStore) -> None:
        self._primary = primary
        self._backup = backup

    def reset(self) -> None:
        for store in (self._primary, self._backup):
            try:
                store.reset()
            except Exception as error:  # noqa: BLE001 - network guard
                print(f"Index reset failed on {type(store).__name__} ({error}).")

    def is_processed(self, video_id: str) -> bool:
        try:
            return self._primary.is_processed(video_id)
        except Exception as error:  # noqa: BLE001 - network guard
            print(f"Primary index read failed ({error}); using local backup.")
            return self._backup.is_processed(video_id)

    def index(
        self,
        asset: VideoAsset,
        transcript: list[TranscriptSegment],
        scripts: list[RenderedScript],
    ) -> None:
        try:
            self._primary.index(asset, transcript, scripts)
        except Exception as error:  # noqa: BLE001 - network guard
            print(f"Primary index write failed ({error}); local backup written.")
        try:
            self._backup.index(asset, transcript, scripts)
        except Exception as error:  # noqa: BLE001 - network guard
            print(f"Local backup write failed ({error}); primary kept.")

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        try:
            return self._primary.search(query, limit)
        except Exception as error:  # noqa: BLE001 - network guard
            print(f"Primary index search failed ({error}); using local backup.")
            return self._backup.search(query, limit)

    def find_similar(
        self, transcript: list[TranscriptSegment], threshold: float, top_k: int = 3
    ) -> SimilarVideo | None:
        try:
            return self._primary.find_similar(transcript, threshold, top_k)
        except Exception as error:  # noqa: BLE001 - network guard
            print(f"Primary similarity search failed ({error}); using local backup.")
            return self._backup.find_similar(transcript, threshold, top_k)


def open_vector_store(
    settings: Settings,
) -> VectorStore | PineconeStore | MirroredStore | None:
    """Return the configured index, mirroring it locally when asked.

    ``REELS_VECTORDB`` selects the primary (``pinecone`` by default, or
    ``chroma``). ``REELS_VECTORDB_BACKUP=chroma`` (default) keeps a local
    Chroma mirror so dedupe and search keep working when the hosted index is
    missing or unreachable; ``REELS_VECTORDB_BACKUP=none`` disables the mirror.
    """
    primary = _open_primary(settings)
    backup: VectorStore | None = None
    if settings.vectordb_backup == "chroma" and settings.vectordb_backend != "chroma":
        try:
            backup = VectorStore(settings)
        except Exception as error:  # noqa: BLE001 - dependency guard
            print(f"Local backup index unavailable ({error}); continuing without it.")
    if primary is None:
        return backup
    if backup is None:
        return primary
    return MirroredStore(primary, backup)


def _open_primary(settings: Settings) -> Any | None:
    if settings.vectordb_backend == "pinecone":
        if not settings.pinecone_api_key:
            print("REELS_VECTORDB=pinecone requires PINECONE_API_KEY; using local backup.")
            return None
        try:
            from .pinecone_store import PineconeStore

            return PineconeStore(settings)
        except Exception as error:  # noqa: BLE001 - network guard
            print(f"Pinecone index unavailable ({error}); using local backup.")
            return None
    if settings.vectordb_backend == "chroma":
        return VectorStore(settings)
    raise RuntimeError(f"Unknown VectorDB provider: {settings.vectordb_backend}")
