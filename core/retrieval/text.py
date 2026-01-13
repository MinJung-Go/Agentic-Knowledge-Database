"""文本召回（关键词匹配）"""
import re
from dataclasses import dataclass
from typing import Callable

import jieba


@dataclass
class TextSearchResult:
    """文本搜索结果"""
    doc_id: str
    chunk_id: str
    content: str
    score: float
    metadata: dict


class TextRetriever:
    """文本召回器

    基于关键词匹配的文本召回，支持：
    - BM25 算法
    - TF-IDF
    - 关键词匹配
    """

    def __init__(self, use_jieba: bool = True):
        self.use_jieba = use_jieba
        self.documents: list[dict] = []
        self.inverted_index: dict[str, list[int]] = {}
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0.0

    def add_documents(self, documents: list[dict], text_key: str = "content"):
        """添加文档到索引"""
        start_idx = len(self.documents)

        for i, doc in enumerate(documents):
            self.documents.append(doc)
            text = doc.get(text_key, "")
            tokens = self._tokenize(text)

            self.doc_lengths.append(len(tokens))

            for token in set(tokens):
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append(start_idx + i)

        total_length = sum(self.doc_lengths)
        self.avg_doc_length = total_length / len(self.doc_lengths) if self.doc_lengths else 0

    def search(
        self,
        query: str,
        top_k: int = 10,
        text_key: str = "content",
    ) -> list[TextSearchResult]:
        """BM25 搜索"""
        query_tokens = self._tokenize(query)

        scores = {}
        k1 = 1.5
        b = 0.75
        N = len(self.documents)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            doc_indices = self.inverted_index[token]
            df = len(doc_indices)
            idf = self._calc_idf(N, df)

            for doc_idx in doc_indices:
                doc = self.documents[doc_idx]
                text = doc.get(text_key, "")
                doc_tokens = self._tokenize(text)
                tf = doc_tokens.count(token)
                doc_length = self.doc_lengths[doc_idx]

                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_length / self.avg_doc_length)
                score = idf * numerator / denominator

                if doc_idx not in scores:
                    scores[doc_idx] = 0
                scores[doc_idx] += score

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_idx, score in sorted_results:
            doc = self.documents[doc_idx]
            results.append(TextSearchResult(
                doc_id=doc.get("doc_id", ""),
                chunk_id=doc.get("chunk_id", ""),
                content=doc.get(text_key, ""),
                score=score,
                metadata=doc.get("metadata", {}),
            ))

        return results

    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        text_key: str = "content",
    ) -> list[TextSearchResult]:
        """关键词匹配搜索"""
        query_tokens = set(self._tokenize(query))

        results = []
        for doc in self.documents:
            text = doc.get(text_key, "")
            doc_tokens = set(self._tokenize(text))

            matched = query_tokens & doc_tokens
            if matched:
                score = len(matched) / len(query_tokens)
                results.append(TextSearchResult(
                    doc_id=doc.get("doc_id", ""),
                    chunk_id=doc.get("chunk_id", ""),
                    content=text,
                    score=score,
                    metadata=doc.get("metadata", {}),
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """分词"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)

        if self.use_jieba:
            tokens = list(jieba.cut(text))
        else:
            tokens = text.split()

        return [t.strip() for t in tokens if t.strip()]

    def _calc_idf(self, N: int, df: int) -> float:
        """计算 IDF"""
        import math
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def clear(self):
        """清空索引"""
        self.documents = []
        self.inverted_index = {}
        self.doc_lengths = []
        self.avg_doc_length = 0.0
