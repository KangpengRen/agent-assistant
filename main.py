from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.assistant.core.budget import MaxTurns, TokenBudget, WallClock
from src.assistant.core.types import AgentLoopOptions, AgentEvent
from src.assistant.core.agent_loop import AgentLoop
from src.assistant.core.planning import PlanThenExecute
from src.assistant.llm.client import DeepSeekClient
from src.assistant.tools.builtin import ListFilesTool, ReadFileTool, WriteFileTool

EXECUTOR_PROMPT = """你是底层执行器，负责完成大计划中的某个特定小步骤。
工作方式：根据步骤目标和完成判据，调用工具完成任务。
【交卷规则】：任务执行完毕（或遇到无法克服的困难）后，**必须调用 `submit_task_result` 工具**来提交你的最终状态。如果不调用此工具直接输出文本，系统将视为你未完成任务并强制报错。

【文件生成铁律】（极其重要）：
除非当前步骤明确要求生成可交付的最终文件（例如任务目标里写了"写入 summary.md"或"覆盖写入"），否则绝对不要在磁盘上创建任何临时记录文件（如 report.md、notes.md 等）。把中间思考、校验结论直接写在 `submit_task_result` 的 summary 字段里交卷即可。严禁乱写临时文件污染工作区！"""


def custom_logger(e: AgentEvent):
    prefix = "      "
    if e.type == "llm_call_start":
        print(f"{prefix}--- 执行器 第 {e.turn} 轮思考 ---")
    elif e.type == "text":
        print(f"{prefix}🤖 执行器说: {e.payload.get('text')[:50]}...") # 截断一下防刷屏
    elif e.type == "tool_call":
        # 过滤掉交卷工具，免得刷屏
        if e.payload.get('name') != 'submit_task_result':
            print(f"{prefix}🛠️ 决定调用工具: {e.payload.get('name')}")
    elif e.type == "tool_result":
        status = "❌" if e.payload.get("is_error") else "✅"
        output = str(e.payload.get("output"))
        print(f"{prefix}👀 工具结果 ({status}): {output[:80]}...")
    elif e.type == "verification_failed":
        print(f"{prefix}⚠️ 外部校验拦截: {e.payload.get('error')}")
    elif e.type == "done":
        print(f"{prefix}✅ 提交成功，当前步骤执行完毕。")
    elif e.type == "aborted":
        print(f"{prefix}🚨 执行器被阻断！(原因: {e.payload.get('reason')})")


def check_summary_file():
    """硬性校验器：专门用来约束 summary.md 的字数"""
    p = Path("summary.md")
    if not p.exists():
        return "你声称完成了，但 summary.md 不存在！请立即重写。"
    content = p.read_text(encoding="utf-8")
    if len(content) > 200:
        return f"文件字数严重超标（当前 {len(content)} 字，限制 180 字）。请立即删减并覆盖原文件！"
    return None


def make_executor(step) -> AgentLoop:
    """ 接收 step 参数，动态决定是否挂载字数校验器"""
    opts = AgentLoopOptions(
        system_prompt=EXECUTOR_PROMPT,
        tools=[ListFilesTool(), ReadFileTool(), WriteFileTool()],
        on_event=custom_logger
    )

    # 动态判断：只有当前任务与 summary.md 相关的写入，才挂载 verifier
    verifier = None
    if "summary.md" in step.goal and ("写" in step.goal or "覆盖" in step.goal or "撰写" in step.goal):
        verifier = check_summary_file

    return AgentLoop(
        opts,
        stop_conditions=[
            MaxTurns(12),
            TokenBudget(50_000),
            WallClock(120)
        ],
        verifier=verifier
    )


def main():
    client = DeepSeekClient()

    p2e_agent = PlanThenExecute(
        llm=client,
        make_executor=make_executor,
        max_replans=2,
        total_budget=0.05
    )

    task = "请先列出当前目录下的所有文件名，然后读取 README.md，最后写一份 200 字以内的 summary.md"

    print(f"\n ==== 启动 Plan-then-Execute ====")
    final_report = p2e_agent.run(task)

    print("\n\n==== 最终交付报告 ====")
    print(final_report)


if __name__ == "__main__":
    main()