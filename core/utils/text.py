import hashlib


def generate_id(content: str) -> str:
    """生成内容哈希ID"""
    return hashlib.md5(content.encode()).hexdigest()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """文本分块"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
