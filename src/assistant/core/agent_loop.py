"""
真正的循环核心。它不负责 HTTP 请求，也不负责具体工具逻辑，而是把 LLM 客户端和工具注册表组合起来，按 ReAct 模式驱动任务执行。
"""

import json
from typing import Any
from .budget import LoopState, StopCondition
from .types import AgentLoopOptions, AgentEvent
from ..llm.client import DeepSeekClient
from ..tools.registry import build_tool_registry, tools_to_wire


class AgentAbortedError(RuntimeError):
    """当预算耗尽导致强制中止时抛出"""
    pass

# 强制模型使用的“交卷”工具
SUBMIT_RESULT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_task_result",
        "description": "任务执行完毕（或遇到无法克服的困难）后，必须调用此工具提交最终状态。",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "failed"]},
                "summary": {"type": "string", "description": "执行结果的最终文本摘要，或失败的详细原因"}
            },
            "required": ["status", "summary"]
        }
    }
}

class AgentLoop:
    def __init__(self, opts: AgentLoopOptions, stop_conditions: list[StopCondition], verifier=None):
        self.opts = opts
        self.stop_conditions = stop_conditions
        self.verifier = verifier
        self.tools_by_name = build_tool_registry(opts.tools)
        self.llm = DeepSeekClient()
        self.total_cost = 0.0

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
            # 1. 预算检查
            for cond in self.stop_conditions:
                d = cond.check(state)
                if d.should_stop:
                    self._emit("aborted", state.turn, reason=d.reason)
                    self.total_cost = state.total_cost
                    final_summary = self._finalize(messages, d.reason)
                    raise AgentAbortedError(final_summary)

            state.turn += 1
            self._emit("llm_call_start", state.turn)

            # 2. 调用 LLM (低推理预算)
            try:
                # 把交卷工具混入原本的工具列表发给大模型
                current_tools = tools_to_wire(self.opts.tools) + [SUBMIT_RESULT_TOOL]
                reply = self.llm.chat(messages, tools=current_tools, reasoning_effort="low")
            except Exception as e:
                self._emit("aborted", state.turn, reason=f"LLM_API_ERROR: {e}")
                self.total_cost = state.total_cost
                raise AgentAbortedError(f"❌ 系统提示：大模型服务连接失败。原因：{e}")

            choice = reply["choices"][0]
            message = choice["message"]
            state.record_usage(reply.get("usage", {}))
            messages.append(message)

            # 3. 输出文本事件 & 拦截截断
            if text := message.get("content"):
                self._emit("text", state.turn, text=text.strip())

            if choice.get("finish_reason") == "length":
                self._emit("aborted", state.turn, reason="max_tokens截断，正在请求精简重试")
                messages.pop()
                messages.append({"role": "user", "content": "输出触发了最大长度限制被硬截断，请精简你的回答。"})
                continue

            # 处理终止与交卷逻辑
            if choice.get("finish_reason") != "tool_calls":
                # 如果大模型不调工具，直接返回纯文本，说明它在“嘴炮”或者想强行结束
                # 只有一种情况允许它结束：它是通过调用了 submit_task_result 结束的（这会在下面的 for 循环里处理）
                self._emit("aborted", state.turn, reason="模型未使用规定工具交卷")
                self.total_cost = state.total_cost
                raise AgentAbortedError("执行器违规：未调用 submit_task_result 工具就擅自结束了任务。")

            # 执行工具调用
            for tool_call in message.get("tool_calls", []):
                name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])

                # 拦截交卷工具
                if name == "submit_task_result":
                    status = args.get("status")
                    summary = args.get("summary")

                    if status == "failed":
                        self._emit("aborted", state.turn, reason=f"模型宣告失败: {summary}")
                        self.total_cost = state.total_cost
                        raise AgentAbortedError(f"执行器宣告失败：{summary}")

                    # 如果状态是 success，再让 verifier 过一眼（防造假）
                    if self.verifier:
                        # 构造一个虚拟的对话历史给 verifier 看
                        temp_msgs = messages + [{"role": "assistant", "content": summary}]
                        if err := self.verifier(temp_msgs):
                            self._emit("verification_failed", state.turn, error=err)
                            # 校验没通过，强行打回！
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": f"【外部系统校验拦截】你声称成功了，但校验失败：{err}。请继续修复。"
                            })
                            break # 跳出工具循环，重新进入大循环让大模型反思

                    # 校验通过，正常退出
                    self._emit("done", state.turn, final_text=summary)
                    self.total_cost = state.total_cost
                    return summary

                # ---- 以下为普通工具的常规处理 ----
                self._emit("tool_call", state.turn, name=name, input=args)

                try:
                    tool = self.tools_by_name.get(name)
                    if not tool:
                        raise ValueError(f"未知工具: {name}")
                    output = tool.execute(args)
                    is_error = False
                except Exception as e:
                    output = (f"工具调用彻底失败: {e}。\n"
                              f"【系统警告】请绝对不要再尝试调用此工具！请立即停止当前思路，检查是否有其他替代方案，或者向用户报告失败。")
                    is_error = True

                self._emit("tool_result", state.turn, name=name, output=output, is_error=is_error)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(output)
                })

    def _finalize(self, messages: list, reason: str) -> str:
        """收尾轮：禁用工具，强制总结"""
        messages.append({
            "role": "user",
            "content": f"运行预算已耗尽（{reason}）。请立即停止行动，总结：已完成事项、未完成事项、下一步建议"
        })
        reply = self.llm.chat(messages, tools=[])
        content = self.llm._exact_content(reply)
        return f"任务因 {reason} 终止\n" + content