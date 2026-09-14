"""CLI entry point for knowledge-base ingestion (see Day 11 plan item 9):

    python -m app.ingest [directory]

Defaults to `KNOWLEDGE_DIRECTORY` (config.yaml: knowledge.directory) if no
directory is given.
"""

import sys
from pathlib import Path

from .config import KNOWLEDGE_DIRECTORY
from .knowledge import ingest_directory


def main() -> None:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else KNOWLEDGE_DIRECTORY
    print(f"Ingesting {directory}...")
    print()

    report = ingest_directory(directory)

    for path, chunk_count in report.per_document_chunks.items():
        print(path)
        print(f"  {chunk_count} chunks")
        print()

    print("Total:")
    print(f"{report.chunks_indexed} chunks indexed")
    print(
        f"{report.documents_indexed} documents indexed, "
        f"{report.documents_skipped} unchanged, "
        f"{report.documents_removed} removed"
    )


if __name__ == "__main__":
    main()
