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
            try:
                reply = self.llm.chat(messages, tools_to_wire(self.opts.tools))
            except Exception as e:
                self._emit("aborted", state.turn, reason=f"LLM_API_ERROR: {e}")
                return f"❌ 系统提示：大模型服务连接失败，Agent已安全停止。原因：{e}"
            choice = reply["choices"][0]
            message = choice["message"]
            state.record_usage(reply.get("usage", {}))
            messages.append(message)

            # 3. 输出文本事件
            if text := message.get("content"):
                self._emit("text", state.turn, text=text.strip())

                # 在这里拦截“长度截断” (max_tokens)，无论是文本截断还是工具JSON截断，finish_reason 都会是 "length"
                # ==========================================================
                if choice.get("finish_reason") == "length":
                    self._emit("aborted", state.turn, reason="max_tokens截断，正在请求精简重试")

                    # 1. 必须把刚才那条残缺的 assistant 消息丢掉！因为它可能带着半截文本，或者解析不出来的半截 JSON。
                    messages.pop()

                    # 2. 伪造一条用户提示，把错误原因作为“观察”回填给模型
                    messages.append({
                        "role": "user",
                        "content": "你刚才的输出触发了最大长度限制被硬截断了。"
                                   "请精简你的回答，或者分多次调用工具。如果你是准备调用工具，请确保参数不要过长。"
                    })

                    # 3. 直接跳过后面的正常终止判断，进入下一轮循环，让大模型重新说
                    continue

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
                    continue

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
                    # 把底层报错“翻译”成带有强制指令的系统警告
                    output = (f"工具调用彻底失败: {e}。\n"
                              f"【系统警告】该工具已在内层重试3次均失败，服务端已宕机或网络断开。"
                              f"请绝对不要再尝试调用此工具！请立即停止任务，并直接向用户报告失败原因。")
                    is_error = True

                self._emit("tool_result", state.turn, name=name, output=output, is_error=is_error)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(output)
                })

    def _finalize(self, messages: list, reason: str) -> str:
        """收尾轮：禁用工具的最后一次调用，把残局翻译成可交接的总结"""
        messages.append({
            "role": "user",
            "content": f"运行预算已耗尽（{reason}）。请立即停止行动，"
                       f"总结：已完成事项、未完成事项、下一步建议"
        })
        # 禁用工具，强制总结
        reply = self.llm.chat(messages, tools=[])
        content = self.llm._exact_content(reply)
        return f"任务因 {reason} 终止\n" + content