"""
定义全书通用的核心接口，后续章节基本只增不改。
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, Optional


class Tool(Protocol):
    """工具接口：描述信息给模型看，execute 在本地执行"""
    name: str
    description: str
    input_schema: dict[str, Any]

    def execute(self, input: dict[str, Any]) -> str: ...


@dataclass
class AgentEvent:
    """生命周期事件，用于 GUI 渲染或可观测性"""
    type: Literal["llm_call_start", "text", "tool_call", "tool_result", "done", "aborted"]
    turn: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentLoopOptions:
    """Agent 循环的配置选项"""
    system_prompt: str
    tools: list[Tool]
    max_turns: int
    on_event: Optional[Callable[[AgentEvent], None]] = None