import asyncio
import operator
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import os
from typing import Annotated
from langchain_core.messages import HumanMessage, RemoveMessage
import httpx
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from typing_extensions import TypedDict

from graph import tools_set, my_prompt, utils

load_dotenv()
EMBED_API_KEY = os.getenv("EMBED_API_KEY")
apiKey = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")
MAX_TOKENS=60000

class State(TypedDict):
    messages: Annotated[list, add_messages]
    #反思记忆（建议覆盖模式：str）
    reflection: str
    #任务规划（覆盖模式：str，由 task_llm 节点生成）
    task_plan: str
    #任务进度（覆盖模式：str，由 reflect_llm 节点更新）
    task_progress: str
    total_tokens: Annotated[int, operator.add]

timeout_config = httpx.Timeout(100.0)

llm = init_chat_model("mimo-v2.5-pro",
                      model_provider="openai",
                      api_key=apiKey,
                      base_url=base_url,
                      timeout=timeout_config,  # 注入超时配置
                      max_retries=10            # 发生超时或网络错误时最多重试10次
                      )

llm_flash = init_chat_model("mimo-v2.5",
                      model_provider="openai",
                      api_key=apiKey,
                      base_url=base_url,
                      timeout=timeout_config,  # 注入超时配置
                      max_retries=10            # 发生超时或网络错误时最多重试10次
                      )


async def memory_manager(state: State):
    messages = state["messages"]

    current_context_length = utils.count_context_tokens(messages)
    print(f"📊 [Debug] 当前上下文 Token: {current_context_length} / {MAX_TOKENS}")

    if current_context_length > MAX_TOKENS:
        print(f"\n🧹 [系统底层] 当前上下文 ({current_context_length} Tokens) 超标，触发记忆压缩...")

        # 1. 修正边界：至少需要保留首条(1) + 待压缩层(>0) + 尾部保留(6) = 8条以上
        if len(messages) <= 7:
            print("⚠️ 警告：消息条数过少，单条消息携带巨量数据，无法安全切片压缩。")
            return {}

        messages_to_compress = messages[1:-6]

        summary_result = await llm_flash.ainvoke([SystemMessage(content=my_prompt.get_summary_prompt(messages_to_compress))])

        target_id = messages_to_compress[0].id

        summary_msg = SystemMessage(
            content=f"【已被压缩的历史背景摘要】：\n{summary_result.content}",
            id=target_id
        )

        delete_commands = [RemoveMessage(id=msg.id) for msg in messages_to_compress[1:] if msg.id]

        return {"messages": delete_commands + [summary_msg]}

    return {}


from pydantic import BaseModel, Field

class ReflectionOutput(BaseModel):
    """反思节点的结构化输出"""
    diagnosis: str = Field(description="诊断结论：如果一切正常填'一切正常'，否则填纠错指令")
    progress: str = Field(default="", description="执行进度：各步骤的完成状态")
    current_status: str = Field(default="", description="当前状态：一句话描述当前在做什么")


tool_node = ToolNode(tools=tools_set.all_tools)
llm_with_tools = llm.bind_tools(tools=tools_set.all_tools)
llm_reflect = llm.with_structured_output(ReflectionOutput)


async def chatbot(state: State):
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

    return {"messages": [message], "reflection": "","total_tokens":current_cost}


async def reflect_llm(state: State):
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
    last_msg = state["messages"][-1]
    # 只在有新 human 消息时才规划
    if last_msg.type != "human":
        return {}

    response = await llm.ainvoke([
        SystemMessage(content=my_prompt.task_prompt),
        HumanMessage(content=last_msg.content)
    ])
    current_cost = utils.get_total_tokens(response)
    print(f"\n[Task Agent 规划结果]\n{response.content}\n")
    return {"task_plan": response.content, "task_progress": "", "total_tokens": current_cost}


graph_builder = StateGraph(State)
graph_builder.add_node("task_llm", task_llm)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tool_node", tool_node)
graph_builder.add_node("reflect_llm", reflect_llm)
graph_builder.add_node("memory_manager", memory_manager)
graph_builder.add_edge(START, "task_llm")
graph_builder.add_edge("task_llm", "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {
        "tools": "tool_node",
        "__end__": END
    }
)
graph_builder.add_edge("tool_node", "memory_manager")
graph_builder.add_edge("memory_manager", "reflect_llm")
graph_builder.add_edge("reflect_llm", "chatbot")

print("你好！很高兴见到你！😊")
print("有什么我可以帮你的吗？无论是日常问题、工作协助还是闲聊，我都很乐意陪你聊聊！")

async def main():
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as memory:
        graph = graph_builder.compile(checkpointer=memory)
        try:
            # 可视化
            img_data = graph.get_graph().draw_mermaid_png()
            with open("graph.png", "wb") as f:
                f.write(img_data)
            print("图形已保存为 graph.png")
        except Exception as e:
            print("可视化失败：", e)
        session_counter=utils.get_JsonCounter()
        config = {"configurable": {"thread_id": f"{session_counter}"}}
        state = await graph.aget_state(config)
        if state and state.values.get("messages"):
            print(f"[恢复] 检测到上次任务，共 {len(state.values['messages'])} 条消息")
            print(f"[恢复] 累计 token: {state.values.get('total_tokens', 0)}")
            # 展示最近几条消息让用户知道上下文
            for msg in state.values["messages"][-4:]:
                print(f"  [{msg.type}] {msg.content[:100]}...")

            choice = input("继续上次任务？(y/n): ")
            if choice != 'y':
                session_counter+=1
                utils.UpdateJsonCounter(session_counter)
                config = {"configurable": {"thread_id": f"{session_counter}"}}
                state = await graph.aget_state(config)
                print("\033[2J\033[H", end="", flush=True)
                print("你好！很高兴见到你！😊")
                print("有什么我可以帮你的吗？无论是日常问题、工作协助还是闲聊，我都很乐意陪你聊聊！")
            else:
                # 用户选了 y，检查图是否还在执行中（中途退出）
                if state.next and state.next != ():
                    print("\n[自动继续] 上次任务中断，正在继续执行...")
                    printed_ids = set()
                    async for event in graph.astream(None, config, stream_mode="values"):
                        for msg in event.get("messages", []):
                            msg_id = getattr(msg, "id", None) or hash(msg.content)
                            if msg_id not in printed_ids:
                                msg.pretty_print()
                                printed_ids.add(msg_id)
                    state = await graph.aget_state(config)
                    total_session_cost = state.values.get("total_tokens", 0)
                    print("\n" + "=" * 40)
                    print(f"本次交互总计消耗 Token: {total_session_cost}")
                    print("=" * 40 + "\n")
        while True:
            if state.interrupts:
                print(state.interrupts[0].value)
            print("（输入完毕后输入END）")
            lines = []
            print(">>")
            while True:
                line = input()
                if line == "END":
                    break
                lines.append(line)
            content = "\n".join(lines)
            if not content:
                continue
            if content == "exit":
                break
            if content == "<<1":
                print("回退到上一步")
                list_state = [s async for s in graph.aget_state_history(config)]
                state_config=None
                for a in list_state:
                    if a.next!=():
                        continue
                    else:
                        state_config=a
                        break
                if len(list_state) >= 1:
                    events = graph.astream(None, state_config.config, stream_mode="values")
                    async for event in events:
                        pass  # 即使不需要在终端重复打印，也必须迭代完毕
                    # 2. 必须重新获取跳跃后的最新状态，覆盖主循环的变量！
                    state = await graph.aget_state(config)
                    continue
            if state.interrupts:
                payload = Command(resume={"data": content})
            else:
                payload = {"messages": [{"role": "user", "content": content}]}
                # 替代原有的 prev_msg_count 逻辑
                printed_ids = set()

                async for event in graph.astream(payload, config, stream_mode="values"):
                    for msg in event["messages"]:
                        # 获取唯一标识（兼容性兜底）
                        msg_id = getattr(msg, "id", None) or hash(msg.content)

                        if msg_id not in printed_ids:
                            msg.pretty_print()
                            printed_ids.add(msg_id)
                            state = await graph.aget_state(config)
                            total_session_cost = state.values.get("total_tokens", 0)
                            print("\n" + "=" * 40)
                            print(f"本次交互总计消耗 Token: {total_session_cost}")
                            print("=" * 40 + "\n")

            state = await graph.aget_state(config)


    # for state in graph.get_state_history(config):
    #     print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
    #     print("-" * 80)

if __name__ == "__main__":
    # 3. 启动异步事件循环
    asyncio.run(main())