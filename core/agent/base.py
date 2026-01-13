from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Agent 基类"""

    name: str = "base"

    def __init__(self):
        AgentRegistry.register(self)

    @abstractmethod
    async def run(self, input_data: Any) -> Any:
        pass


class AgentRegistry:
    """Agent 注册中心"""

    _agents: dict[str, "BaseAgent"] = {}

    @classmethod
    def register(cls, agent: "BaseAgent"):
        cls._agents[agent.name] = agent

    @classmethod
    def get(cls, name: str) -> "BaseAgent | None":
        return cls._agents.get(name)

    @classmethod
    def list_agents(cls) -> list[str]:
        return list(cls._agents.keys())
