import json
from dataclasses import dataclass
from typing import Literal


@dataclass
class Step:
    id: int
    goal: str
    done_when: str
    status: Literal["pending", "done", "failed"] = "pending"
    result: str = ""


PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "提交任务执行计划。每步必须有可验证的完成判据；一般 3-8 步。",
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "goal": {"type": "string", "description": "要执行的具体动作"},
                            "done_when": {"type": "string", "description": "判断该步完成的客观标准"}
                        },
                        "required": ["goal", "done_when"]
                    }
                }
            },
            "required": ["steps"]
        }
    }
}


class PlanThenExecute:
    def __init__(self, llm, make_executor, max_replans: int = 2, total_budget: float = 5.0):
        self.llm = llm
        self.make_executor = make_executor
        self.max_replans = max_replans
        self.total_budget = total_budget
        self.accumulated_cost = 0.0

    def plan(self, task: str, context: str = "") -> list[Step]:
        prompt = f"任务：{task}\n{context}\n请仔细规划执行步骤。"
        # 全局规划：高推理预算
        reply = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=[PLAN_TOOL],
            reasoning_effort="high"
        )

        message = reply["choices"][0]["message"]
        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            raise ValueError("Planner 未返回结构化计划")

        steps_data = json.loads(tool_calls[0]["function"]["arguments"]).get("steps", [])
        return [Step(i, s["goal"], s["done_when"]) for i, s in enumerate(steps_data, 1)]

    def run(self, task: str) -> str:
        print(f"\n[Planner] 正在拆解任务: {task}")
        steps = self.plan(task)
        for s in steps:
            print(f"  -> 步骤 {s.id}: {s.goal} (完成判据: {s.done_when})")

        replans = 0
        while pending := [s for s in steps if s.status == "pending"]:
            if self.accumulated_cost >= self.total_budget:
                print(f"🚨 总预算 ({self.total_budget:.4f} USD) 已耗尽，强制停止！")
                return self._abort_summary(steps)

            step = pending[0]
            done_ctx = "\n".join(f"步骤{s.id}[{s.goal}]：{s.result}" for s in steps if s.status == "done")

            print(f"\n[Executor] 开始执行步骤 {step.id}: {step.goal}")
            executor = self.make_executor()
            try:
                # 执行器只看到"已完成结论 + 当前步骤"
                step.result = executor.run(
                    f"已完成进展：\n{done_ctx or '（无）'}\n\n"
                    f"当前步骤：{step.goal}\n完成判据：{step.done_when}"
                )
                step.status = "done"
                print(f"✅ 步骤 {step.id} 完成")
            except Exception as e:
                step.status = "failed"
                step.result = str(e)
                print(f"❌ 步骤 {step.id} 失败: {e}")
            finally:
                # 无论成功失败，把花掉的钱记账
                self.accumulated_cost += executor.total_cost

            # 触发增量重规划
            if any(s.status == "failed" for s in steps):
                if replans >= self.max_replans:
                    print(f"🚨 达到最大重规划次数 ({self.max_replans})，终止任务。")
                    return self._abort_summary(steps)

                replans += 1
                print(f"\n[Replanner] 触发增量重规划 (第 {replans} 次)...")
                steps = self._replan(task, steps)
                for s in steps:
                    if s.status != "done":
                        print(f"  -> 新步骤 {s.id}: {s.goal}")

        return "\n\n".join(f"## 步骤 {s.id}: {s.goal}\n{s.result}" for s in steps if s.status == "done")

    def _replan(self, task: str, steps: list[Step]) -> list[Step]:
        done_steps = [s for s in steps if s.status == "done"]
        report = "\n".join(f"[{s.status}] {s.goal}：{s.result[:200]}" for s in steps)

        fresh_steps = self.plan(task,
                                context=f"既有进展与失败记录：\n{report}\n【警告】只规划剩余工作，不要重复已完成的步骤！")

        new_steps = done_steps + [Step(len(done_steps) + i, s.goal, s.done_when) for i, s in enumerate(fresh_steps, 1)]
        return new_steps

    def _abort_summary(self, steps: list[Step]) -> str:
        return ("[任务中止]\n" + "\n".join(f"[{s.status}] {s.goal}：{s.result[:200]}" for s in steps))