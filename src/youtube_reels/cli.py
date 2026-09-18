from __future__ import annotations

import argparse

from .config import Settings
from .pipeline import AlreadyProcessed, DuplicateContent, ProcessError, process
from .vectordb import open_vector_store


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YouTube to original shorts: audio → transcription → rewrite → video."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    process_parser = commands.add_parser("process", help="Download, rewrite and generate shorts.")
    process_parser.add_argument("url")
    process_parser.add_argument("--max-clips", type=int, default=3)
    process_parser.add_argument("--force-transcribe", action="store_true")
    process_parser.add_argument(
        "--force", action="store_true", help="Reprocess even if the video is already indexed."
    )
    search_parser = commands.add_parser("search", help="Search already-processed videos.")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=5)
    commands.add_parser(
        "reset-index", help="Discard all indexed videos (Pinecone + Chroma mirror)."
    )
    arguments = parser.parse_args()
    settings = Settings.from_env()
    if arguments.command == "process":
        try:
            outputs = process(
                arguments.url,
                settings,
                arguments.max_clips,
                arguments.force_transcribe,
                arguments.force,
            )
        except (AlreadyProcessed, DuplicateContent) as error:
            print(error)
            return
        except ProcessError as error:
            print(f"Error: {error}")
            return
        print("Shorts created (original rewritten content):")
        print(*(str(path) for path in outputs), sep="\n")
    elif arguments.command == "reset-index":
        store = open_vector_store(settings)
        if store is None:
            print("No index configured (REELS_VECTORDB=chroma or pinecone).")
            return
        store.reset()
        print("Index reset. Next process run rebuilds it (Pinecone + Chroma mirror).")
    else:
        store = open_vector_store(settings)
        if store is None:
            print("No index configured (REELS_VECTORDB=chroma or pinecone).")
            return
        for result in store.search(arguments.query, arguments.limit):
            metadata = result["metadata"]
            kind = metadata.get("kind", "?")
            if kind == "script":
                label = metadata.get("hook") or "short"
                tag = "REWRITTEN"
            else:
                label = f"{metadata.get('start', 0):.0f}s"
                tag = "SOURCE"
            print(f"[{metadata['video_id']} {tag}] {label}: {result['text']}")


if __name__ == "__main__":
    main()