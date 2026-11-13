import time
from dataclasses import dataclass, field
from typing import Protocol

@dataclass
class LoopState:
    turn: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def record_usage(self, usage: dict) -> None:
        # 兼容 DeepSeek/OpenAI 和 Claude 的字段名
        self.input_tokens += usage.get("prompt_tokens", usage.get("input_tokens", 0))
        self.output_tokens += usage.get("completion_tokens", usage.get("output_tokens", 0))

@dataclass
class StopDecision:
    should_stop: bool
    reason: str = ""

class StopCondition(Protocol):
    def check(self, state: LoopState) -> StopDecision: ...

@dataclass
class MaxTurns:
    limit: int
    def check(self, s: LoopState) -> StopDecision:
        return StopDecision(s.turn >= self.limit, f"max_turns({s.turn}/{self.limit})")

@dataclass
class TokenBudget:
    limit: int
    def check(self, s: LoopState) -> StopDecision:
        used = s.input_tokens + s.output_tokens
        return StopDecision(used >= self.limit, f"token_budget({used}/{self.limit})")

@dataclass
class CostBudget:
    limit_used: float
    price_in: float = 5.0
    price_out: float = 25.0
    def check(self, s: LoopState) -> StopDecision:
        # 注意：DeepSeek 的 API 计费单位通常是百万 tokens（此处仅为示例逻辑）
        cost = (s.input_tokens * self.price_in + s.output_tokens * self.price_out) / 1e6
        return StopDecision(cost >= self.limit_used, f"cost_budget(${cost:.4f}/${self.limit_used})")

@dataclass
class WallClock:
    limit_seconds: float
    def check(self, s: LoopState) -> StopDecision:
        elapsed = time.monotonic() - s.started_at
        return StopDecision(elapsed >= self.limit_seconds, f"wall_clock({elapsed:.0f}s/{self.limit_seconds:.0f}s)")