import hashlib
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .models import Document

__all__ = ["load_documents"]

# V1 supports only Markdown and plain text (see Day 11 plan item 1).
_SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}

# Frontmatter keys mapped onto Document's metadata fields (see Day 12 plan item 13).
_METADATA_KEYS = ("source_type", "document_type", "project", "owner_id")


def _split_frontmatter(content: str) -> tuple[dict, str]:
    """Splits an optional leading `---\\n...\\n---` YAML block off `content`. Returns
    (metadata, body). Missing or malformed frontmatter yields ({}, content unchanged) -
    metadata is opt-in, not required (see Day 12 plan item 13)."""
    if not content.startswith("---\n") and not content.startswith("---\r\n"):
        return {}, content

    end = content.find("\n---", 4)
    if end == -1:
        return {}, content

    raw_frontmatter = content[4:end]
    body_start = content.find("\n", end + 1)
    body = content[body_start + 1 :] if body_start != -1 else ""

    try:
        parsed = yaml.safe_load(raw_frontmatter)
    except yaml.YAMLError:
        return {}, content
    if not isinstance(parsed, dict):
        return {}, content

    metadata = {key: parsed[key] for key in _METADATA_KEYS if key in parsed}
    return metadata, body


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
    items 1-3, 10). An optional YAML frontmatter block is parsed for metadata
    (source_type/document_type/project/owner_id) and stripped from the body used for
    chunking/embedding (see Day 12 plan item 13). The content hash covers the raw
    file text, so editing frontmatter alone still triggers re-ingestion."""
    directory = Path(directory)
    if not directory.exists():
        return []

    documents = []
    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue

        raw_content = file_path.read_text(encoding="utf-8", errors="ignore")
        content_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
        metadata, body = _split_frontmatter(raw_content)
        relative_path = file_path.relative_to(directory).as_posix()
        stat = file_path.stat()

        documents.append(
            Document(
                id=relative_path,
                path=relative_path,
                title=_title_from_content(body, fallback=file_path.stem),
                content=body,
                content_hash=content_hash,
                created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
                updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                source_type=metadata.get("source_type"),
                document_type=metadata.get("document_type"),
                project=metadata.get("project"),
                owner_id=metadata.get("owner_id"),
            )
        )
    return documents
