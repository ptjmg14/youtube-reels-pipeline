from __future__ import annotations

from typing import Any

from .config import Settings
from .dedupe import jaccard, shingles, transcript_chunks, transcript_text
from .models import RenderedScript, SimilarVideo, TranscriptSegment, VideoAsset

_DIMENSION = 1024  # multilingual-e5-large
_EMBEDDING_BATCH = 96
_UPSERT_BATCH = 100
_METADATA_BYTES = 35000  # Pinecone caps metadata at 40 KB per vector


class PineconeStore:
    """Serverless index with server-side embeddings (no local ONNX/Torch).

    Vectors live in Pinecone and are embedded on their API. The free Starter
    plan holds ~100k vectors with 5M embedding tokens per month — roughly 330
    videos at ~15k tokens each — and removes the ~10s ONNX startup plus the
    local embedding pass from every run (free, no credit card).
    """

    def __init__(self, settings: Settings) -> None:
        try:
            from pinecone import Pinecone
        except ImportError as error:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Install the hosted index: pip install -e '.[pinecone]'"
            ) from error
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is required for this backend.")
        self._model = settings.pinecone_embedding_model
        self._client = Pinecone(api_key=settings.pinecone_api_key)
        
        # Check if index exists, create if not
        # Note: pinecone v10 structure for index creation might differ
        # Assuming index management as per newer API
        try:
            self._client.describe_index(settings.pinecone_index_name)
        except Exception as error:  # noqa: BLE001 - create the index when missing
            print(f"Index {settings.pinecone_index_name} not found ({error}); creating.")
            self._client.create_index(
                name=settings.pinecone_index_name,
                dimension=_DIMENSION,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": settings.pinecone_region}},
            )
        self._index_name = settings.pinecone_index_name
        self._index = self._client.Index(settings.pinecone_index_name)

    def reset(self) -> None:
        """Delete the hosted index; the store rebuilds it on next init."""
        self._client.delete_index(self._index_name)

    def is_processed(self, video_id: str) -> bool:
        fetched = self._index.fetch(ids=[f"{video_id}:transcript:0"])
        return bool(fetched.vectors)

    def index(
        self,
        asset: VideoAsset,
        transcript: list[TranscriptSegment],
        scripts: list[RenderedScript],
    ) -> None:
        texts: list[str] = []
        records: list[dict[str, Any]] = []
        for index, segment in enumerate(transcript):
            texts.append(segment.text)
            records.append(
                {
                    "id": f"{asset.video_id}:transcript:{index}",
                    "text": segment.text,
                    "video_id": asset.video_id,
                    "title": asset.title,
                    "kind": "transcript",
                    "start": segment.start,
                    "end": segment.end,
                }
            )
        for script in scripts:
            texts.append(script.narration)
            records.append(
                {
                    "id": f"{asset.video_id}:script:{script.number}",
                    "text": script.narration,
                    "video_id": asset.video_id,
                    "title": asset.title,
                    "kind": "script",
                    "hook": script.hook,
                    "clip": script.number,
                }
            )
        full_text = transcript_text(transcript)
        if full_text:
            stored_text = full_text.encode("utf-8")[:_METADATA_BYTES].decode("utf-8", "ignore")
            texts.append(full_text)
            records.append(
                {
                    "id": f"{asset.video_id}:transcript:full",
                    "text": stored_text,
                    "video_id": asset.video_id,
                    "title": asset.title,
                    "kind": "transcript_full",
                }
            )
        if not texts:
            return
        
        # Pinecone v10 inference with batching
        all_vectors = []
        for i in range(0, len(texts), _EMBEDDING_BATCH):
            batch_texts = texts[i : i + _EMBEDDING_BATCH]
            batch_records = records[i : i + _EMBEDDING_BATCH]
            
            response = self._client.inference.embed(
                model=self._model,
                inputs=batch_texts,
                parameters={"input_type": "passage", "truncate": "END"},
            )
            
            all_vectors.extend([
                {
                    "id": record["id"],
                    "values": embedding["values"],
                    "metadata": {key: value for key, value in record.items() if key != "id"},
                }
                for record, embedding in zip(batch_records, response, strict=True)
            ])
            
        for start in range(0, len(all_vectors), _UPSERT_BATCH):
            self._index.upsert(vectors=all_vectors[start : start + _UPSERT_BATCH])

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        response = self._client.inference.embed(
            model=self._model,
            inputs=[query],
            parameters={"input_type": "query", "truncate": "END"},
        )
        matches = self._index.query(
            vector=response[0]["values"],
            top_k=limit,
            include_metadata=True,
            filter={"kind": {"$ne": "transcript_full"}},
        )
        return [
            {
                "text": match.metadata.get("text", ""),
                "metadata": match.metadata,
                "distance": match.score,
            }
            for match in matches.matches
        ]

    def find_similar(
        self, transcript: list[TranscriptSegment], threshold: float, top_k: int = 3
    ) -> SimilarVideo | None:
        """Recall candidate videos semantically, then confirm exact overlap.

        Candidates are verified with character n-gram Jaccard against the
        stored full transcript, so "same topic" videos are not flagged as
        duplicates of "same content" videos.
        """
        chunks = transcript_chunks(transcript)
        if not chunks:
            return None
        response = self._client.inference.embed(
            model=self._model,
            inputs=chunks,
            parameters={"input_type": "query", "truncate": "END"},
        )
        candidate_ids: list[str] = []
        for embedding in response:
            matches = self._index.query(
                vector=embedding["values"], top_k=top_k, include_metadata=True
            )
            for match in matches.matches:
                video_id = match.metadata.get("video_id")
                if video_id and video_id not in candidate_ids:
                    candidate_ids.append(video_id)
        query_shingles = shingles(" ".join(chunks))
        best: SimilarVideo | None = None
        for video_id in candidate_ids:
            record_id = f"{video_id}:transcript:full"
            fetched = self._index.fetch(ids=[record_id])
            if record_id not in fetched.vectors:
                continue
            metadata = fetched.vectors[record_id].metadata or {}
            stored_text = metadata.get("text", "")
            if not stored_text:
                continue
            score = jaccard(query_shingles, shingles(stored_text))
            if best is None or score > best.score:
                best = SimilarVideo(
                    video_id=video_id,
                    title=metadata.get("title", ""),
                    score=score,
                )
        return best if best and best.score >= threshold else None
