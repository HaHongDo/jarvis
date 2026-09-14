import hashlib
from datetime import datetime, timezone
from pathlib import Path

from .models import Document

__all__ = ["load_documents"]

# V1 supports only Markdown and plain text (see Day 11 plan item 1).
_SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}


def _title_from_content(content: str, fallback: str) -> str:
    """Uses the first Markdown heading as the title if present, otherwise falls
    back to the file name."""
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
        return fallback
    return fallback


def load_documents(directory: Path) -> list[Document]:
    """Walks `directory` recursively for Markdown/plain-text files and loads each as
    a `Document`, hashing its content for incremental ingestion (see Day 11 plan
    items 1-3, 10)."""
    directory = Path(directory)
    if not directory.exists():
        return []

    documents = []
    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue

        content = file_path.read_text(encoding="utf-8", errors="ignore")
        relative_path = file_path.relative_to(directory).as_posix()
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        stat = file_path.stat()

        documents.append(
            Document(
                id=relative_path,
                path=relative_path,
                title=_title_from_content(content, fallback=file_path.stem),
                content=content,
                content_hash=content_hash,
                created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
                updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            )
        )
    return documents
