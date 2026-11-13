"""
把工具实现从主流程里抽出来，每个工具一个函数，并附上给模型看的 Schema。
"""

from pathlib import Path
from typing import Any
from ..core.types import Tool


class ListFilesTool(Tool):
    name = "list_files"
    description = "列出目录下的所有文件名，用于了解工作区里有什么"
    input_schema = {
        "type": "object",
        "properties": {"dir": {"type": "string", "description": "目录路径，缺省为当前目录"}},
        "required": []
    }

    def execute(self, input: dict[str, Any]) -> str:
        return "\n".join(sorted(p.name for p in Path(input.get("dir", ".")).iterdir()))


class ReadFileTool(Tool):
    name = "read_file"
    description = "读取文件内容（截断到 20k 字符）"
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "文件路径"}},
        "required": ["path"]
    }

    def execute(self, input: dict[str, Any]) -> str:
        return Path(input["path"]).read_text(encoding="utf-8")[:20_000]


class WriteFileTool(Tool):
    name = "write_file"
    description = "写入文件内容（覆盖原文件），用于产出报告等交付物"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径"},
            "content": {"type": "string", "description": "要写入的内容"}
        },
        "required": ["path", "content"]
    }

    def execute(self, input: dict[str, Any]) -> str:
        Path(input["path"]).write_text(input["content"], encoding="utf-8")
        return f"文件已写入 {input['path']} ({len(input['content'])} 字符)"