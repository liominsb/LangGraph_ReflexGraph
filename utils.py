import os

import tiktoken
from langchain_core.messages import BaseMessage
import json


# 【关键修复】增加了 agent_type: str = "主Agent"
def get_total_tokens(message: BaseMessage) -> int:
    if getattr(message, 'type', '') != "ai":
        return 0

    # 1. 优先尝试新版 LangChain 标准格式
    if hasattr(message, 'usage_metadata') and message.usage_metadata:
        return message.usage_metadata.get('total_tokens', 0)

    # 2. 兜底旧版或特定模型的嵌套格式
    if hasattr(message, 'response_metadata') and message.response_metadata:
        usage = message.response_metadata.get('token_usage') or message.response_metadata.get('usage') or {}
        return usage.get('total_tokens', 0)

    return 0


def count_context_tokens(messages: list[BaseMessage], model_name: str = "gpt-4o") -> int:
    """本地极速估算当前上下文的 Token 长度（包含工具调用参数）"""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for msg in messages:
        # 1. 统计基础文本内容
        if msg.content:
            num_tokens += len(encoding.encode(str(msg.content)))

        # 2. 【核心修复】必须统计大模型发出的工具调用参数
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            # 将工具调用的字典序列化为字符串进行长度估算
            tool_calls_str = json.dumps(msg.tool_calls, ensure_ascii=False)
            num_tokens += len(encoding.encode(tool_calls_str))

    return num_tokens

def get_json_counter() -> int:
    """读取会话计数器"""
    state_file = "counter.json"
    if os.path.exists(state_file):
        with open(state_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            session_counter = data["counter"]
    else:
        session_counter = 1
    return session_counter


# 保持向后兼容
get_JsonCounter = get_json_counter

def update_json_counter(session_counter: int) -> None:
    """更新会话计数器"""
    state_file = "counter.json"
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump({"counter": session_counter}, f)


# 保持向后兼容
UpdateJsonCounter = update_json_counter