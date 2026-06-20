import os

from langchain_core.messages import BaseMessage
import time

# 【关键修复】增加了 agent_type: str = "主Agent"
def parse_and_print_token_usage(message: BaseMessage, task_name: str = "默认任务", agent_type: str = "主Agent") -> dict:
    """
    解析 LangChain 消息对象中的 Token 消耗与缓存命中率，并打印统计信息。
    同时将消耗追加到本地 CSV 账单中。

    Args:
        message (BaseMessage): 当前流式输出或生成的最后一条消息对象。
        task_name (str): 当前执行的任务名称或提问内容，用于账单记录区分。
        agent_type (str): 执行任务的 Agent 角色类型。默认值为"主Agent"。

    Returns:
        dict: 包含 Token 统计明细的字典。如果不是 AI 消息则返回空字典。
    """
    if getattr(message, 'type', '') != "ai":
        return {}

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    cached_tokens = 0

    # 1. 优先尝试解析 LangChain 统一的 usage_metadata (新版支持)
    if hasattr(message, 'usage_metadata') and message.usage_metadata:
        usage = message.usage_metadata
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)

        # 提取缓存读取量 (增加 None 安全检查)
        input_details = usage.get('input_token_details') or {}
        cached_tokens = input_details.get('cache_read', 0)

    # 2. 备选方案：从原始响应格式中提取
    elif hasattr(message, 'response_metadata') and message.response_metadata:
        meta = message.response_metadata
        usage = meta.get('token_usage') or meta.get('usage') or {}

        if usage:
            input_tokens = input_tokens or usage.get('prompt_tokens', 0)
            output_tokens = output_tokens or usage.get('completion_tokens', 0)
            total_tokens = total_tokens or usage.get('total_tokens', 0)

            # 提取缓存数据 (增加 None 安全检查)
            prompt_details = usage.get('prompt_tokens_details') or {}
            cached_tokens = cached_tokens or prompt_details.get('cached_tokens', 0)

    # 3. 计算命中率并格式化输出
    if input_tokens > 0:
        hit_rate = (cached_tokens / input_tokens) * 100
        print(f"📊 [Token消耗统计 | {agent_type} - {task_name[:10]}]")
        print(f"   ├─ 输入: {input_tokens} (其中命中缓存: {cached_tokens}, 命中率: {hit_rate:.2f}%)")
        print(f"   ├─ 输出: {output_tokens}")
        print(f"   └─ 总计: {total_tokens}")

        # ========== 极简账单记录 (多维细化写入带表头) ==========
        try:
            csv_filename = "my_token_costs.csv"
            # 检查文件是否已经存在
            file_exists = os.path.exists(csv_filename)

            with open(csv_filename, "a", encoding="utf-8") as f:
                # 如果文件是刚创建的，第一行先写入标题栏
                if not file_exists:
                    f.write("时间,角色,任务,原始输入消耗,命中缓存量,输出消耗\n")

                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                clean_task = str(task_name)[:20].replace(',', '，').replace('\n', '')

                # 正常写入具体数据
                f.write(f"{current_time},{agent_type},{clean_task},{input_tokens},{cached_tokens},{output_tokens}\n")
        except Exception as e:
            print(f"⚠️ 账单写入失败: {e}")
        # =================================

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens
    }