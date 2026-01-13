"""TextRetriever 单元测试"""
import pytest

from core.retrieval.text import TextRetriever, TextSearchResult


class TestTextSearchResult:
    """TextSearchResult 数据类测试"""

    def test_creation(self):
        """测试创建"""
        result = TextSearchResult(
            doc_id="doc_001",
            chunk_id="chunk_001",
            content="测试内容",
            score=0.85,
            metadata={"filename": "test.pdf"},
        )
        assert result.doc_id == "doc_001"
        assert result.chunk_id == "chunk_001"
        assert result.content == "测试内容"
        assert result.score == 0.85
        assert result.metadata["filename"] == "test.pdf"


class TestTextRetriever:
    """TextRetriever 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        retriever = TextRetriever()
        assert retriever.use_jieba is True
        assert len(retriever.documents) == 0
        assert len(retriever.inverted_index) == 0

    def test_init_without_jieba(self):
        """测试不使用 jieba"""
        retriever = TextRetriever(use_jieba=False)
        assert retriever.use_jieba is False

    def test_add_documents(self, sample_documents):
        """测试添加文档"""
        retriever = TextRetriever()
        retriever.add_documents(sample_documents)

        assert len(retriever.documents) == 3
        assert len(retriever.doc_lengths) == 3
        assert retriever.avg_doc_length > 0

    def test_add_documents_builds_inverted_index(self, sample_documents):
        """测试添加文档构建倒排索引"""
        retriever = TextRetriever()
        retriever.add_documents(sample_documents)

        # 倒排索引应该包含一些词
        assert len(retriever.inverted_index) > 0

    def test_search_basic(self, sample_documents):
        """测试基本搜索"""
        retriever = TextRetriever()
        retriever.add_documents(sample_documents)

        results = retriever.search("人工智能", top_k=2)

        assert len(results) > 0
        assert all(isinstance(r, TextSearchResult) for r in results)
        # 结果应该按分数降序排列
        if len(results) > 1:
            assert results[0].score >= results[1].score

    def test_search_no_results(self, sample_documents):
        """测试无匹配结果"""
        retriever = TextRetriever()
        retriever.add_documents(sample_documents)

        # 使用不会被 jieba 分词匹配到的查询
        results = retriever.search("zzzyyyxxx123456")

        assert len(results) == 0

    def test_search_top_k(self, sample_documents):
        """测试 top_k 参数"""
        retriever = TextRetriever()
        retriever.add_documents(sample_documents)

        results = retriever.search("学习", top_k=1)

        assert len(results) <= 1

    def test_keyword_search(self, sample_documents):
        """测试关键词搜索"""
        retriever = TextRetriever()
        retriever.add_documents(sample_documents)

        results = retriever.keyword_search("机器学习", top_k=3)

        assert len(results) > 0
        # 分数应该在 0-1 之间
        for result in results:
            assert 0 <= result.score <= 1

    def test_keyword_search_no_results(self, sample_documents):
        """测试关键词搜索无结果"""
        retriever = TextRetriever()
        retriever.add_documents(sample_documents)

        results = retriever.keyword_search("xyz123abc")

        assert len(results) == 0

    def test_tokenize_chinese(self):
        """测试中文分词"""
        retriever = TextRetriever(use_jieba=True)
        tokens = retriever._tokenize("人工智能是未来发展的方向")

        assert len(tokens) > 1
        assert "人工智能" in tokens or "人工" in tokens

    def test_tokenize_english(self):
        """测试英文分词"""
        retriever = TextRetriever(use_jieba=False)
        tokens = retriever._tokenize("Hello World Test")

        assert len(tokens) == 3
        assert "hello" in tokens  # 应该被转为小写

    def test_tokenize_removes_punctuation(self):
        """测试分词移除标点"""
        retriever = TextRetriever(use_jieba=False)
        tokens = retriever._tokenize("Hello, World! Test.")

        assert "," not in tokens
        assert "!" not in tokens
        assert "." not in tokens

    def test_calc_idf(self):
        """测试 IDF 计算"""
        retriever = TextRetriever()

        # N=100, df=10 应该有正的 IDF
        idf = retriever._calc_idf(100, 10)
        assert idf > 0

        # df 越大，IDF 越小
        idf_high_df = retriever._calc_idf(100, 50)
        assert idf > idf_high_df

    def test_clear(self, sample_documents):
        """测试清空索引"""
        retriever = TextRetriever()
        retriever.add_documents(sample_documents)

        retriever.clear()

        assert len(retriever.documents) == 0
        assert len(retriever.inverted_index) == 0
        assert len(retriever.doc_lengths) == 0
        assert retriever.avg_doc_length == 0.0

    def test_bm25_scoring(self):
        """测试 BM25 评分"""
        retriever = TextRetriever()
        documents = [
            {"doc_id": "1", "chunk_id": "1", "content": "机器学习是人工智能的核心", "metadata": {}},
            {"doc_id": "2", "chunk_id": "2", "content": "深度学习是机器学习的分支", "metadata": {}},
            {"doc_id": "3", "chunk_id": "3", "content": "自然语言处理应用广泛", "metadata": {}},
        ]
        retriever.add_documents(documents)

        results = retriever.search("机器学习")

        # 包含 "机器学习" 的文档应该排在前面
        assert len(results) >= 2
        assert "机器" in results[0].content or "学习" in results[0].content

    def test_search_with_custom_text_key(self):
        """测试自定义文本字段"""
        retriever = TextRetriever()
        documents = [
            {"doc_id": "1", "chunk_id": "1", "text": "测试文本内容", "metadata": {}},
        ]
        retriever.add_documents(documents, text_key="text")

        results = retriever.search("测试", text_key="text")

        assert len(results) >= 1

    def test_incremental_add_documents(self):
        """测试增量添加文档"""
        retriever = TextRetriever()

        # 第一批文档
        docs1 = [{"doc_id": "1", "chunk_id": "1", "content": "第一批文档", "metadata": {}}]
        retriever.add_documents(docs1)
        assert len(retriever.documents) == 1

        # 第二批文档
        docs2 = [{"doc_id": "2", "chunk_id": "2", "content": "第二批文档", "metadata": {}}]
        retriever.add_documents(docs2)
        assert len(retriever.documents) == 2

        # 搜索应该能找到两批文档
        results = retriever.search("文档")
        assert len(results) == 2
