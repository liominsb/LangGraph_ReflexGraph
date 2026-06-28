from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langgraph.prebuilt import ToolNode

from graph import config, my_prompt, utils, tools_set
from graph.state import State, ReflectionOutput

# ── 工具 / LLM 绑定 ──
tool_node = ToolNode(tools=tools_set.all_tools)
llm_with_tools = config.llm.bind_tools(tools=tools_set.all_tools)
llm_reflect = config.llm.with_structured_output(ReflectionOutput)


async def memory_manager(state: State):
    """上下文窗口管理：超过 MAX_TOKENS 时压缩历史消息"""
    messages = state["messages"]

    current_context_length = utils.count_context_tokens(messages)
    print(f"📊 [Debug] 当前上下文 Token: {current_context_length} / {config.MAX_TOKENS}")

    if current_context_length > config.MAX_TOKENS:
        print(f"\n🧹 [系统底层] 当前上下文 ({current_context_length} Tokens) 超标，触发记忆压缩...")

        # 1. 修正边界：至少需要保留首条(1) + 待压缩层(>0) + 尾部保留(6) = 8条以上
        if len(messages) <= 7:
            print("⚠️ 警告：消息条数过少，单条消息携带巨量数据，无法安全切片压缩。")
            return {}

        messages_to_compress = messages[1:-6]

        summary_result = await config.llm_flash.ainvoke([SystemMessage(content=my_prompt.get_summary_prompt(messages_to_compress))])

        target_id = messages_to_compress[0].id

        summary_msg = SystemMessage(
            content=f"【已被压缩的历史背景摘要】：\n{summary_result.content}",
            id=target_id
        )

        delete_commands = [RemoveMessage(id=msg.id) for msg in messages_to_compress[1:] if msg.id]

        return {"messages": delete_commands + [summary_msg]}

    return {}


async def chatbot(state: State):
    """主代理节点：注入系统提示、任务规划、反思纠错"""
    sys_msg = SystemMessage(content=my_prompt.one_prompt)
    full_messages = [sys_msg] + state["messages"]

    # 注入任务规划
    task_plan = state.get("task_plan", "")
    if task_plan:
        plan_msg = SystemMessage(
            content=f"【任务规划】\n{task_plan}\n\n请严格按照以上计划执行。"
        )
        full_messages.insert(1, plan_msg)

    # 注入任务进度
    task_progress = state.get("task_progress", "")
    if task_progress:
        progress_msg = SystemMessage(
            content=f"【任务进度】\n{task_progress}"
        )
        full_messages.insert(2, progress_msg)

    reflection_content = state.get("reflection", "")
    if reflection_content:
        # 将字符串包装为高优先级的系统消息
        reflection_msg = SystemMessage(
            content=f"【最高优先级纠错指令】\n必须严格遵循以下指示调整你的行动规划：\n{reflection_content}"
        )
        # 挂载到最后，确保 LLM 第一时间看到
        full_messages.append(reflection_msg)

    message = await llm_with_tools.ainvoke(full_messages)
    current_cost = utils.get_total_tokens(message)

    return {"messages": [message], "reflection": "", "total_tokens": current_cost}


async def reflect_llm(state: State):
    """反思/诊断节点：检查最近执行轨迹，输出结构化诊断"""
    # 1. 提取最近消息
    recent_messages = state["messages"][-6:]

    trajectory_lines = []
    for msg in recent_messages:
        if msg.type == "human":
            trajectory_lines.append(f"用户提问: {msg.content}")
        elif msg.type == "ai":
            if msg.tool_calls:
                trajectory_lines.append(f"Agent尝试调用工具: {msg.tool_calls}")
            else:
                trajectory_lines.append(f"Agent回复: {msg.content}")
        elif msg.type == "tool":
            trajectory_lines.append(f"工具返回结果: {msg.content}")

    trajectory_str = "\n".join(trajectory_lines)

    # 2. 构建 prompt
    task_plan = state.get("task_plan", "")
    task_progress = state.get("task_progress", "")

    sys_msg = SystemMessage(content=my_prompt.reflect_prompt)

    if task_plan:
        human_msg = HumanMessage(
            content=f"""【任务规划】
{task_plan}

【上次进度】
{task_progress if task_progress else "尚未开始"}

【最近执行轨迹】
{trajectory_str}"""
        )
    else:
        human_msg = HumanMessage(
            content=f"【执行轨迹记录】\n{trajectory_str}"
        )

    # 3. 结构化输出
    response: ReflectionOutput = await llm_reflect.ainvoke([sys_msg, human_msg])
    print(f"\n[Critic 反思结果] => 诊断: {response.diagnosis}")

    current_cost = 0  # with_structured_output 可能不返回 token 信息，兜底

    # 4. 构建返回值
    result = {"total_tokens": current_cost}

    if response.diagnosis == "一切正常":
        result["reflection"] = ""
    else:
        result["reflection"] = response.diagnosis

    # 5. 有任务规划时更新进度
    if task_plan:
        progress_text = ""
        if response.progress:
            progress_text += response.progress
        if response.current_status:
            progress_text += f"\n\n【当前状态】\n{response.current_status}"
        result["task_progress"] = progress_text

    return result


async def task_llm(state: State):
    """任务规划节点：对新的 human 消息生成结构化任务计划"""
    last_msg = state["messages"][-1]
    # 只在有新 human 消息时才规划
    if last_msg.type != "human":
        return {}

    response = await config.llm.ainvoke([
        SystemMessage(content=my_prompt.task_prompt),
        HumanMessage(content=last_msg.content)
    ])
    current_cost = utils.get_total_tokens(response)
    print(f"\n[Task Agent 规划结果]\n{response.content}\n")
    return {"task_plan": response.content, "task_progress": "", "total_tokens": current_cost}
