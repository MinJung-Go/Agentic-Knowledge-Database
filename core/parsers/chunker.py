"""文本分块策略"""
import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Chunk:
    """文本块"""
    content: str
    index: int
    metadata: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.content)


class TextChunker:
    """文本分块器

    支持多种分块策略：
    - 固定长度分块
    - 句子分块
    - 段落分块
    - 语义分块（递归）
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "]

    def chunk_text(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """分块文本（递归分割）"""
        if metadata is None:
            metadata = {}

        chunks = self._recursive_split(text, self.separators)

        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {**metadata, "chunk_index": i}
            result.append(Chunk(content=chunk_text, index=i, metadata=chunk_metadata))

        return result

    def chunk_by_sentences(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """按句子分块"""
        if metadata is None:
            metadata = {}

        sentence_endings = re.compile(r'([。！？.!?]+)')
        parts = sentence_endings.split(text)

        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sentence = parts[i] + parts[i + 1]
            sentences.append(sentence.strip())
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())

        chunks = self._merge_small_chunks(sentences)

        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {**metadata, "chunk_index": i}
            result.append(Chunk(content=chunk_text, index=i, metadata=chunk_metadata))

        return result

    def chunk_by_paragraphs(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """按段落分块"""
        if metadata is None:
            metadata = {}

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = self._merge_small_chunks(paragraphs)

        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {**metadata, "chunk_index": i}
            result.append(Chunk(content=chunk_text, index=i, metadata=chunk_metadata))

        return result

    def chunk_fixed(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """固定长度分块"""
        if metadata is None:
            metadata = {}

        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunk_metadata = {**metadata, "chunk_index": index}
            chunks.append(Chunk(content=chunk_text, index=index, metadata=chunk_metadata))

            start = end - self.chunk_overlap
            index += 1

        return chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """递归分割文本"""
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            return self._split_by_length(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        parts = text.split(separator)

        chunks = []
        current_chunk = ""

        for part in parts:
            test_chunk = current_chunk + separator + part if current_chunk else part

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                if len(part) > self.chunk_size:
                    sub_chunks = self._recursive_split(part, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        return self._add_overlap(chunks)

    def _split_by_length(self, text: str) -> list[str]:
        """按长度强制分割"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap

        return chunks

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """合并过小的块"""
        if not chunks:
            return []

        result = []
        current = chunks[0]

        for chunk in chunks[1:]:
            if len(current) + len(chunk) + 1 <= self.chunk_size:
                current = current + "\n" + chunk
            else:
                result.append(current)
                current = chunk

        if current:
            result.append(current)

        return result

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """添加块之间的重叠"""
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]

            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
            new_chunk = overlap_text + curr_chunk

            if len(new_chunk) <= self.chunk_size * 1.2:
                result.append(new_chunk)
            else:
                result.append(curr_chunk)

        return result


class MarkdownChunker(TextChunker):
    """Markdown 文档分块器"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        separators = [
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n\n",
            "\n",
            "。",
            ".",
            " ",
        ]
        super().__init__(chunk_size, chunk_overlap, separators)

    def chunk_by_headers(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """按标题分块"""
        if metadata is None:
            metadata = {}

        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

        sections = []
        last_end = 0
        current_headers = []

        for match in header_pattern.finditer(text):
            if last_end < match.start():
                content = text[last_end:match.start()].strip()
                if content:
                    sections.append({
                        "headers": list(current_headers),
                        "content": content,
                    })

            level = len(match.group(1))
            title = match.group(2)

            current_headers = [h for h in current_headers if h["level"] < level]
            current_headers.append({"level": level, "title": title})

            last_end = match.end()

        if last_end < len(text):
            content = text[last_end:].strip()
            if content:
                sections.append({
                    "headers": list(current_headers),
                    "content": content,
                })

        result = []
        for i, section in enumerate(sections):
            header_path = " > ".join(h["title"] for h in section["headers"])
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "headers": header_path,
            }
            result.append(Chunk(
                content=section["content"],
                index=i,
                metadata=chunk_metadata,
            ))

        return result
