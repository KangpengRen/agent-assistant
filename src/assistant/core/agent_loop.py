"""
真正的循环核心。它不负责 HTTP 请求，也不负责具体工具逻辑，而是把 LLM 客户端和工具注册表组合起来，按 ReAct 模式驱动任务执行。
"""

import json
from typing import Any
from .budget import LoopState, StopCondition
from .types import AgentLoopOptions, AgentEvent
from ..llm.client import DeepSeekClient
from ..tools.registry import build_tool_registry, tools_to_wire


class AgentLoop:
    def __init__(self, opts: AgentLoopOptions, stop_conditions: list[StopCondition], verifier=None):
        self.opts = opts
        self.stop_conditions = stop_conditions
        self.verifier = verifier
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
        state = LoopState()

        while True:
            # 1. 预算检查 (熔断机制)
            for cond in self.stop_conditions:
                d = cond.check(state)
                if d.should_stop:
                    self._emit("aborted", state.turn, reason=d.reason)
                    return self._finalize(messages, d.reason)

            state.turn += 1
            self._emit("llm_call_start", state.turn)

            # 2. 调用 LLM
            reply = self.llm.chat(messages, tools_to_wire(self.opts.tools))
            choice = reply["choices"][0]
            message = choice["message"]
            state.record_usage(reply.get("usage", {}))
            messages.append(message)

            # 3. 输出文本事件
            if text := message.get("content"):
                self._emit("text", state.turn, text=text.strip())

            # 4. 终止判断与外部校验 (Verifier)
            if choice.get("finish_reason") != "tool_calls":
                final_text = self.llm._exact_content(reply)
                # 如果有外部校验器，且校验返回了错误信息
                if self.verifier and (err := self.verifier(messages)):
                    self._emit("verification_failed", state.turn, error=err)
                    messages.append({
                        "role": "user",
                        "content": f"任务未通过外部校验：{err}。请根据要求修正后继续。"
                    })
                    continue  # 【关键修复】不要 return，继续循环让大模型重新生成！

                # 校验通过，正常退出
                self._emit("done", state.turn, final_text=final_text)
                return final_text

            # 5. 执行工具调用
            for tool_call in message.get("tool_calls", []):
                name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])
                self._emit("tool_call", state.turn, name=name, input=args)

                try:
                    tool = self.tools_by_name.get(name)
                    if not tool:
                        raise ValueError(f"未知工具: {name}")
                    output = tool.execute(args)
                    is_error = False
                except Exception as e:
                    output = f"工具调用失败: {e}"
                    is_error = True

                self._emit("tool_result", state.turn, name=name, output=output, is_error=is_error)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(output)
                })

    def _finalize(self, messages: list, reason: str) -> str:
        """收尾轮：禁用工具的最后一次调用，把残局翻译成可交接的总结"""
        messages.append({ # 【关键修复】拼写 apppend -> append
            "role": "user",
            "content": f"运行预算已耗尽（{reason}）。请立即停止行动，"
                       f"总结：已完成事项、未完成事项、下一步建议"
        })
        # 禁用工具，强制总结
        reply = self.llm.chat(messages, tools=[])
        content = self.llm._exact_content(reply)
        return f"任务因 {reason} 终止\n" + content