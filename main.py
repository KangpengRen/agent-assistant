import json
from dotenv import load_dotenv

load_dotenv()

from src.assistant.core.types import AgentLoopOptions
from src.assistant.core.agent_loop import AgentLoop, AgentEvent
from src.assistant.tools.builtin import ListFilesTool, ReadFileTool, WriteFileTool

SYSTEM_PROMPT = """你是示例助手，一个企业内部任务执行Agent。
工作方式：先思考需要哪些信息，再调用工具获取，信息足够后完成任务，直接给出最终答复。
约束：只操作当前工作目录；文件不存在或读取失败时如实报告，绝不编造内容。"""

def custom_logger(e: AgentEvent):
    if e.type == "llm_call_start":
        print(f"\n--- 第 {e.turn} 轮思考开始 ---")

    elif e.type == "text":
        # payload 是一个字典，从中取出 text
        print(f"🤖 模型说: {e.payload.get('text')}")

    elif e.type == "tool_call":
        tool_name = e.payload.get("name")
        tool_input = e.payload.get("input")
        print(f"🛠️ 决定调用工具: {tool_name}")
        print(f"   参数: {json.dumps(tool_input, ensure_ascii=False)}")

    elif e.type == "tool_result":
        output = str(e.payload.get("output"))
        is_error = e.payload.get("is_error")
        status = "❌ 失败" if is_error else "✅ 成功"
        # 截断输出，防止刷屏
        print(f"👀 工具执行结果 ({status}): {output[:150]}...")

    elif e.type == "done":
        print(f"\n✅ 第 {e.turn} 轮结束，任务完成。")

    elif e.type == "aborted":
        reason = e.payload.get("reason", "未知原因")
        print(f"\n🚨 任务被强制中止！(原因: {reason})，当前轮数: {e.turn}")

def main():
    tools = [ListFilesTool(), ReadFileTool(), WriteFileTool()]
    opts = AgentLoopOptions(
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        max_turns=10,
        on_event=custom_logger
    )

    agent = AgentLoop(opts)
    task = "请列出当前目录下的所有文件名，然后写一份200字以内的 summary.md 总结一下"
    final_answer = agent.run(task)
    print("\n==== 最终答复 ====")
    print(final_answer)


if __name__ == "__main__":
    main()