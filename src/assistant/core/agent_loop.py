"""
真正的循环核心。它不负责 HTTP 请求，也不负责具体工具逻辑，而是把 LLM 客户端和工具注册表组合起来，按 ReAct 模式驱动任务执行。
"""

import json
from typing import Any
from .types import AgentLoopOptions, AgentEvent
from ..llm.client import DeepSeekClient
from ..tools.registry import build_tool_registry, tools_to_wire


class AgentLoop:
    def __init__(self, opts: AgentLoopOptions):
        self.opts = opts
        self.tools_by_name = build_tool_registry(opts.tools)
        self.llm = DeepSeekClient()

    def _emit(self, event_type: str, turn: int, **payload: Any):
        if self.opts.on_event:
            self.opts.on_event(AgentEvent(type=event_type, turn=turn, payload=payload))

    def run(self, task: str) -> str:
        messages = [
            {"role": "system", "content": self.opts.system_prompt},
            {"role": "user", "content": task}
        ]

        for turn in range(1, self.opts.max_turns + 1):
            self._emit("llm_call_start", turn)

            # 1. 调用 LLM
            reply = self.llm.chat(messages, tools_to_wire(self.opts.tools))
            choice = reply["choices"][0]
            message = choice["message"]
            messages.append(message)

            # 2. 输出文本事件
            if text := message.get("content"):
                self._emit("text", turn, text=text.strip())

            # 3. 终止判断
            if choice.get("finish_reason") != "tool_calls":
                self._emit("done", turn, final_text=message.get("content", ""))
                return message.get("content", "")

            # 4. 执行工具调用
            for tool_call in message.get("tool_calls", []):
                name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])
                self._emit("tool_call", turn, name=name, input=args)

                try:
                    tool = self.tools_by_name.get(name)
                    if not tool:
                        raise ValueError(f"未知工具: {name}")
                    output = tool.execute(args)
                    is_error = False
                except Exception as e:
                    output = f"工具调用失败: {e}"
                    is_error = True

                self._emit("tool_result", turn, name=name, output=output, is_error=is_error)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(output)
                })

        self._emit("aborted", self.opts.max_turns, reason="max_turns")
        raise RuntimeError(f"达到 {self.opts.max_turns} 轮上限，未完成任务")