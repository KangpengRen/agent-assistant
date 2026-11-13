"""
封装 HTTP 请求，把重试、超时、认证等细节都收拢在这里。
它直接对接 DeepSeek 的 Chat Completions 接口。
"""

import os
import requests
from typing import Any

class DeepSeekClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("缺少环境变量 DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/chat/completions"

    def chat(self, messages: list[dict], tools: list[dict]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "tools": tools,
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        # 简单的重试机制（生产环境可改为指数退避）
        for attempt in range(3):
            try:
                resp = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code < 500:
                    raise RuntimeError(f"API {resp.status_code}: {resp.text}")
            except requests.RequestException as e:
                if attempt == 2:
                    raise
        raise RuntimeError("API 请求重试耗尽")

    @staticmethod
    def _exact_content(reply: dict[str, Any]) -> str:
        return reply["choices"][0]["message"].get("content", "")