from abc import ABC, abstractmethod


class BaseParser(ABC):
    """解析器基类"""

    @abstractmethod
    def parse(self, file_path: str) -> str:
        pass
