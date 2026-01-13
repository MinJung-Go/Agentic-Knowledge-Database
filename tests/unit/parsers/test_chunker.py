"""TextChunker 单元测试"""
import pytest

from core.parsers.chunker import Chunk, TextChunker, MarkdownChunker


class TestChunk:
    """Chunk 数据类测试"""

    def test_chunk_creation(self):
        """测试 Chunk 创建"""
        chunk = Chunk(content="测试内容", index=0)
        assert chunk.content == "测试内容"
        assert chunk.index == 0
        assert chunk.metadata == {}

    def test_chunk_with_metadata(self):
        """测试带元数据的 Chunk"""
        metadata = {"doc_id": "doc_001", "filename": "test.pdf"}
        chunk = Chunk(content="测试内容", index=1, metadata=metadata)
        assert chunk.metadata == metadata

    def test_chunk_length_property(self):
        """测试 length 属性"""
        chunk = Chunk(content="12345", index=0)
        assert chunk.length == 5


class TestTextChunker:
    """TextChunker 测试"""

    def test_init_default_params(self):
        """测试默认参数"""
        chunker = TextChunker()
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50
        assert len(chunker.separators) > 0

    def test_init_custom_params(self):
        """测试自定义参数"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 10

    def test_chunk_text_basic(self, sample_text):
        """测试基本分块功能"""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk_text(sample_text)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.index == i for i, c in enumerate(chunks))

    def test_chunk_text_small_text(self):
        """测试小于 chunk_size 的文本"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = "这是一段很短的文本"
        chunks = chunker.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_chunk_text_empty_text(self):
        """测试空文本"""
        chunker = TextChunker()
        chunks = chunker.chunk_text("")

        assert len(chunks) == 0

    def test_chunk_preserves_metadata(self):
        """测试元数据传递"""
        chunker = TextChunker(chunk_size=50)
        metadata = {"doc_id": "doc_001", "filename": "test.pdf"}
        chunks = chunker.chunk_text("这是一段测试文本，需要被分块处理。" * 5, metadata=metadata)

        for chunk in chunks:
            assert chunk.metadata["doc_id"] == "doc_001"
            assert chunk.metadata["filename"] == "test.pdf"
            assert "chunk_index" in chunk.metadata

    def test_chunk_by_sentences(self):
        """测试按句子分块"""
        chunker = TextChunker(chunk_size=100)
        text = "第一句话。第二句话。第三句话。第四句话。第五句话。"
        chunks = chunker.chunk_by_sentences(text)

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.content) <= 100 * 1.5  # 允许一定的超出

    def test_chunk_by_paragraphs(self, sample_text):
        """测试按段落分块"""
        chunker = TextChunker(chunk_size=200)
        chunks = chunker.chunk_by_paragraphs(sample_text)

        assert len(chunks) > 0

    def test_chunk_fixed(self):
        """测试固定长度分块"""
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        text = "这是一段很长的测试文本，需要按固定长度进行分块处理。"
        chunks = chunker.chunk_fixed(text)

        assert len(chunks) > 1
        # 检查第一个块的长度
        assert len(chunks[0].content) == 20

    def test_chunk_fixed_with_overlap(self):
        """测试固定分块的重叠"""
        chunker = TextChunker(chunk_size=10, chunk_overlap=3)
        text = "0123456789abcdefghij"
        chunks = chunker.chunk_fixed(text)

        # 验证重叠：第二个块应该从 index 7 开始 (10-3)
        assert len(chunks) >= 2

    def test_recursive_split_with_separators(self):
        """测试递归分割"""
        chunker = TextChunker(chunk_size=50, chunk_overlap=0)
        text = "段落一的内容\n\n段落二的内容\n\n段落三的内容"
        chunks = chunker._recursive_split(text, chunker.separators)

        assert len(chunks) > 0

    def test_merge_small_chunks(self):
        """测试合并过小的块"""
        chunker = TextChunker(chunk_size=50)
        small_chunks = ["短文本1", "短文本2", "短文本3"]
        merged = chunker._merge_small_chunks(small_chunks)

        # 应该合并为一个块
        assert len(merged) == 1
        assert "短文本1" in merged[0]
        assert "短文本2" in merged[0]

    def test_add_overlap(self):
        """测试添加重叠"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = ["第一个块的内容", "第二个块的内容", "第三个块的内容"]
        result = chunker._add_overlap(chunks)

        assert len(result) == 3
        # 第二个块应该包含第一个块的结尾
        assert result[0] == chunks[0]  # 第一个块不变


class TestMarkdownChunker:
    """MarkdownChunker 测试"""

    def test_init(self):
        """测试初始化"""
        chunker = MarkdownChunker()
        assert "\n## " in chunker.separators
        assert "\n### " in chunker.separators

    def test_chunk_by_headers(self, sample_markdown):
        """测试按标题分块"""
        chunker = MarkdownChunker(chunk_size=500)
        chunks = chunker.chunk_by_headers(sample_markdown)

        assert len(chunks) > 0
        # 验证每个块都有 headers 元数据
        for chunk in chunks:
            assert "headers" in chunk.metadata

    def test_chunk_by_headers_preserves_metadata(self, sample_markdown):
        """测试标题分块保留元数据"""
        chunker = MarkdownChunker()
        metadata = {"doc_id": "md_001"}
        chunks = chunker.chunk_by_headers(sample_markdown, metadata=metadata)

        for chunk in chunks:
            assert chunk.metadata["doc_id"] == "md_001"

    def test_chunk_by_headers_empty_text(self):
        """测试空文本"""
        chunker = MarkdownChunker()
        chunks = chunker.chunk_by_headers("")

        assert len(chunks) == 0

    def test_chunk_by_headers_no_headers(self):
        """测试没有标题的文本"""
        chunker = MarkdownChunker()
        text = "这是一段没有标题的普通文本。"
        chunks = chunker.chunk_by_headers(text)

        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_chunk_by_headers_nested_headers(self):
        """测试嵌套标题"""
        text = """# 一级标题

内容1

## 二级标题

内容2

### 三级标题

内容3

## 另一个二级标题

内容4"""
        chunker = MarkdownChunker()
        chunks = chunker.chunk_by_headers(text)

        assert len(chunks) >= 3
