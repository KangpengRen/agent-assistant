"""
封装 HTTP 请求，把重试、超时、认证等细节都收拢在这里。
它直接对接 DeepSeek 的 Chat Completions 接口。
"""

import os
import requests
from typing import Any
import time

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

                # 成功直接返回
                if resp.status_code == 200:
                    return resp.json()

                # 如果是 429 限流 或 5xx 服务端错误，进行指数退避重试
                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(2 ** attempt)  # 第1次等1秒，第2次等2秒，第3次等4秒
                    continue

                # 其他客户端错误（如 401 鉴权失败、400 参数错误），重试也没用，直接抛出
                raise RuntimeError(f"不可重试的 API 错误 {resp.status_code}: {resp.text}")

            except requests.RequestException as e:
                if attempt == 2:
                    raise RuntimeError(f"网络请求彻底失败: {e}")
                time.sleep(1)
        raise RuntimeError("API 请求重试耗尽")

    @staticmethod
    def _exact_content(reply: dict[str, Any]) -> str:
        return reply["choices"][0]["message"].get("content", "")