from dotenv import load_dotenv

load_dotenv()

from src.assistant.core.budget import MaxTurns, TokenBudget, CostBudget, WallClock
from src.assistant.core.types import AgentLoopOptions, AgentEvent
from src.assistant.core.agent_loop import AgentLoop
from src.assistant.core.planning import PlanThenExecute  # 导入 P2E
from src.assistant.llm.client import DeepSeekClient
from src.assistant.tools.builtin import ListFilesTool, ReadFileTool, WriteFileTool

EXECUTOR_PROMPT = """你是底层执行器，当前只负责完成大计划中的**某一个特定小步骤**。
工作方式：根据给定的步骤目标和完成判据，调用工具完成任务。
约束：
1. 只操作当前工作目录。
2. 不要进行超出当前步骤目标的无意义探索。
3. 完成后，直接输出该步骤的最终结果。**不要输出思考过程、操作记录或完成判据核对**。
"""


def custom_logger(e: AgentEvent):
    prefix = "      "
    if e.type == "llm_call_start":
        print(f"{prefix}--- 执行器 第 {e.turn} 轮思考 ---")
    elif e.type == "text":
        print(f"{prefix}🤖 执行器说: {e.payload.get('text')[:50]}...") # 截断一下防刷屏
    elif e.type == "tool_call":
        print(f"{prefix}🛠️ 决定调用工具: {e.payload.get('name')}")
    elif e.type == "tool_result":
        output = str(e.payload.get("output"))
        is_error = e.payload.get("is_error")
        status = "❌" if is_error else "✅"
        print(f"{prefix}👀 工具结果 ({status}): {output[:100]}...")
    elif e.type == "done":
        print(f"{prefix}✅ 当前步骤执行完毕。")
    elif e.type == "aborted":
        print(f"{prefix}🚨 执行器中止！(原因: {e.payload.get('reason')})")


def make_executor() -> AgentLoop:
    opts = AgentLoopOptions(
        system_prompt=EXECUTOR_PROMPT,
        tools=[ListFilesTool(), ReadFileTool(), WriteFileTool()],
        on_event=custom_logger
    )
    return AgentLoop(
        opts,
        stop_conditions=[
            MaxTurns(5),  # 防死循环
            TokenBudget(50_000),  # 单步上下文防爆
            WallClock(120)  # 单步防卡死
            # 注意：不在这里放 CostBudget，交予外层统筹
        ],
        verifier=None
    )


def main():
    client = DeepSeekClient()

    p2e_agent = PlanThenExecute(
        llm=client,
        make_executor=make_executor,
        max_replans=2,
        total_budget=0.05  # 【全局控制】整个任务最多花 0.05 美元
    )

    task = "请先列出当前目录下的所有文件名，然后读取 README.md，最后写一份 200 字以内的 summary.md"

    print(f"\n ==== 启动 Plan-then-Execute ====")
    final_report = p2e_agent.run(task)

    print("\n\n==== 最终交付报告 ====")
    print(final_report)


if __name__ == "__main__":
    main()