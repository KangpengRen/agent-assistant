"""
维护工具注册表，并提供“将 Tool 协议对象转换为 DeepSeek/OpenAI 所需 tools 格式”的方法。
"""

from typing import Any
from ..core.types import Tool

def build_tool_registry(tools: list[Tool]) -> dict[str, Tool]:
    """构建名字到工具实例的映射"""
    return {t.name: t for t in tools}

def tools_to_wire(tools: list[Tool]) -> list[dict[str, Any]]:
    """转换为 DeepSeek/OpenAI 兼容的 tools 数组"""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema
            }
        }
        for t in tools
    ]