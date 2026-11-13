"""
minimal_agent.py - 最小 ReAct Agent (DeepSeek版)
依赖：仅 requests（HTTP客户端），无任何框架
运行：python minimal_agent.py "任务"
"""
import json
import os
import sys
from pathlib import Path
from typing import Any
import requests
from dotenv import load_dotenv

# 读取 .env 文件中的环境变量
load_dotenv()


# ----------------- 1. 定义本地工具 -----------------

def list_files(inp: dict) -> str:
    return "\n".join(sorted(p.name for p in Path(inp.get("dir", ".")).iterdir()))


def read_file(inp: dict) -> str:
    return Path(inp["path"]).read_text(encoding="utf-8")[:20_000]


def write_file(inp: dict) -> str:
    Path(inp["path"]).write_text(inp["content"], encoding="utf-8")
    return f"文件已写入 {inp['path']} ({len(inp['content'])} 字符)"


# 工具清单（提供给大模型看）
TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_files",
        "description": "列出目录下的所有文件名，用于了解工作区里有什么",
        "input_schema": {
            "type": "object",
            "properties": {"dir": {"type": "string", "description": "目录路径，缺省为当前目录"}},
            "required": [],
        },
        "run": list_files,
    },
    {
        "name": "read_file",
        "description": "读取文件内容（截断到 20k 字符）",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "文件路径"}},
            "required": ["path"],
        },
        "run": read_file,
    },
    {
        "name": "write_file",
        "description": "写入文件内容（覆盖原文件），用于产出报告等交付物",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "文件路径"},
                           "content": {"type": "string", "description": "要写入的内容"}},
            "required": ["path", "content"],
        },
        "run": write_file,
    }
]

# 方便后面通过名字快速查找工具
TOOLS_BY_NAME: dict[str, Any] = {t["name"]: t for t in TOOLS}

SYSTEM_PROMPT = """
你是示例助手，一个企业内部任务执行Agent。
工作方式：先思考需要哪些信息，再调用工具获取，信息足够后完成任务，直接给出最终答复。
约束：只操作当前工作目录；文件不存在或读取失败时如实报告，绝不编造内容。
"""

# ----------------- 2. 调用大模型 -----------------

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
url = "https://api.deepseek.com/chat/completions"


def call_llm(messages: list) -> dict:
    if not API_KEY:
        raise ValueError("请检查 .env 文件中是否配置了 DEEPSEEK_API_KEY")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # 将本地工具格式转换为 OpenAI/DeepSeek 要求的格式
    openai_tools = []
    for t in TOOLS:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"]
            }
        })

    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "tools": openai_tools,
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 1024
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"API {response.status_code}: {response.text}")
    return response.json()


# ----------------- 3. 运行 Agent 逻辑 -----------------

MAX_TURNS = 10  # 防止死循环的硬刹车


def run_agent(task: str) -> str:
    # 初始化消息
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task}
    ]

    for turn in range(MAX_TURNS):
        reply = call_llm(messages)

        choice = reply["choices"][0]
        message = choice["message"]

        # 将模型的回复（可能包含 tool_calls）加入历史
        messages.append(message)

        # 打印模型的文字回复
        if message.get("content"):
            print(f"\n[思考] {message['content'].strip()}")

        # 如果模型不调用工具了，说明给出了最终答复
        if choice.get("finish_reason") != "tool_calls":
            return message.get("content", "")

        # 执行工具调用
        for t in message.get("tool_calls", []):
            tool_name = t["function"]["name"]
            tool_args = json.loads(t["function"]["arguments"])

            print(f"[行动] {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

            try:
                # 真正的本地执行代码
                content = TOOLS_BY_NAME[tool_name]["run"](tool_args)
            except Exception as e:
                content = f"工具调用失败：{e}"

            print(f"[观察] {str(content)[:200]}...")

            # 将工具执行结果回传给模型
            messages.append({
                "role": "tool",
                "tool_call_id": t["id"],
                "content": str(content)
            })

    raise RuntimeError(f"达到 {MAX_TURNS} 轮上限，未完成任务——硬中止生效")


# ----------------- 4. 入口 -----------------

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "请列出当前目录下的所有文件名，然后写一份200字以内的 summary.md 总结一下"
    print(f"\n ==== 开始执行任务 ====\n")
    print(f"\n ==== 最终答复 ====\n{run_agent(task)}")