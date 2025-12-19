"""Simple character-based text chunking."""

import re
from dataclasses import dataclass
from typing import Optional

from .config import CHUNK_SIZE, CHUNK_OVERLAP

# ~4 chars per token
CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    text: str
    token_count: int  # Estimated
    chunk_index: int
    start_char: int
    end_char: int
    page_number: Optional[int] = None


class TextChunker:
    """Chunks text into overlapping segments using character-based splitting."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size_chars = chunk_size * CHARS_PER_TOKEN
        self.chunk_overlap_chars = chunk_overlap * CHARS_PER_TOKEN
        self.page_pattern = re.compile(r'\[PAGE (\d+)\]')

    def chunk_text(self, text: str) -> list[Chunk]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []

        total_chars = len(text)

        # Single chunk if small enough
        if total_chars <= self.chunk_size_chars:
            return [Chunk(
                text=text,
                token_count=total_chars // CHARS_PER_TOKEN,
                chunk_index=0,
                start_char=0,
                end_char=total_chars,
                page_number=self._extract_page_number(text)
            )]

        chunks = []
        chunk_index = 0
        i = 0

        while i < total_chars:
            end = min(i + self.chunk_size_chars, total_chars)
            chunk_text = text[i:end]

            chunks.append(Chunk(
                text=chunk_text,
                token_count=len(chunk_text) // CHARS_PER_TOKEN,
                chunk_index=chunk_index,
                start_char=i,
                end_char=end,
                page_number=self._extract_page_number(chunk_text)
            ))

            chunk_index += 1

            # Break if we've reached the end
            if end >= total_chars:
                break

            # Advance with overlap
            i = end - self.chunk_overlap_chars

        return chunks

    def _extract_page_number(self, text: str) -> Optional[int]:
        """Extract the first page number from text if present."""
        match = self.page_pattern.search(text)
        if match:
            return int(match.group(1))
        return None


def chunk_document(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Chunk a document into overlapping segments."""
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    return chunker.chunk_text(text)
