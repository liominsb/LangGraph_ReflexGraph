import asyncio
import os
from typing import Annotated

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

from graph import tools_set, my_prompt, utils, csv_Visualization

load_dotenv()
EMBED_API_KEY = os.getenv("EMBED_API_KEY")
apiKey = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    #反思记忆（建议覆盖模式：str）
    reflection: str

llm = init_chat_model("mimo-v2.5",
                      model_provider="openai",
                      api_key=apiKey,
                      base_url=base_url,
                      )

tool_node = ToolNode(tools=tools_set.all_tools)
llm_with_tools = llm.bind_tools(tools=tools_set.all_tools)


async def chatbot(state: State):
    sys_msg = SystemMessage(content=my_prompt.one_prompt)
    full_messages = [sys_msg] + state["messages"]

    reflection_content = state.get("reflection", "")
    if reflection_content:
        # 将字符串包装为高优先级的系统消息
        reflection_msg = SystemMessage(
            content=f"【最高优先级纠错指令】\n必须严格遵循以下指示调整你的行动规划：\n{reflection_content}"
        )
        # 挂载到最后，确保 LLM 第一时间看到
        full_messages.append(reflection_msg)
    # 使用 ainvoke 替代 invoke
    message = await llm_with_tools.ainvoke(full_messages)
    return {"messages": [message], "reflection": ""}

def interpret_llm(state: State):
    sys_msg = SystemMessage(content=my_prompt.interpret_bot_prompt)
    full_messages = [sys_msg] + state["messages"]
    message = llm.invoke(full_messages)
    return {"messages": [message]}


from langchain_core.messages import HumanMessage


async def reflect_llm(state: State):
    # 1. 提取原生对象
    recent_messages = state["messages"][-6:]

    # 2. 【核心修复】：将 Message 对象降维打包成纯文本格式的“诊断日志”
    # 这样模型看到的是一份“案卷”，而不是正在进行的对话
    trajectory_lines = []
    for msg in recent_messages:
        # 根据不同消息类型提取可读内容
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

    sys_msg = SystemMessage(content=my_prompt.reflect_prompt)
    human_msg = HumanMessage(
        content=f"请你作为独立的审查员，诊断以下执行轨迹记录，严格按照你的系统设定输出结论。\n\n【执行轨迹记录】\n{trajectory_str}"
    )

    # 4. 执行反思
    response = await llm.ainvoke([sys_msg, human_msg])
    print("\n[Critic 反思结果] =>", response.content)

    # 5. 状态更新
    if "一切正常" in response.content:
        return {"reflection": ""}
    return {"reflection": response.content}

memory = MemorySaver()

graph_builder = StateGraph(State)
graph_builder.add_node("interpret_llm", interpret_llm)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tool_node", tool_node)
graph_builder.add_node("reflect_llm", reflect_llm)
graph_builder.add_edge(START, "interpret_llm")
graph_builder.add_edge("interpret_llm", "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {
        "tools": "tool_node",
        "__end__": END
    }
)
graph_builder.add_edge("tool_node", "reflect_llm")
graph_builder.add_edge("reflect_llm", "chatbot")
graph = graph_builder.compile(checkpointer=memory)
try:
# 可视化
    img_data = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(img_data)
    print("图形已保存为 graph.png")
except Exception as e:
    print("可视化失败：", e)

print("你好！很高兴见到你！😊")
print("有什么我可以帮你的吗？无论是日常问题、工作协助还是闲聊，我都很乐意陪你聊聊！")

async def main():
    config = {"configurable": {"thread_id": "1"}}
    state = graph.get_state(config)
    while True:
        if state.interrupts:
            print(state.interrupts[0].value)
        content = input(">")
        if content == "exit":
            break
        if content == "<<1":
            print("回退到上一步")
            list_state = list(graph.get_state_history(config))
            state_config=None
            for a in list_state:
                if a.next!=():
                    continue
                else:
                    state_config=a
                    break
            if len(list_state) >= 1:
                events = graph.stream(None, state_config.config, stream_mode="values")
                for event in events:
                    pass  # 即使不需要在终端重复打印，也必须迭代完毕
                # 2. 必须重新获取跳跃后的最新状态，覆盖主循环的变量！
                state = graph.get_state(config)
                continue
        if state.interrupts:
            payload = Command(resume={"data": content})
        else:
            payload = {"messages": [{"role": "user", "content": content}]}
        async for event in graph.astream(payload, config, stream_mode="values"):
            event["messages"][-1].pretty_print()
            utils.parse_and_print_token_usage(event["messages"][-1], task_name=content, agent_type="主Agent")
        state = graph.get_state(config)

    # for state in graph.get_state_history(config):
    #     print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
    #     print("-" * 80)

if __name__ == "__main__":
    # 3. 启动异步事件循环
    asyncio.run(main())
    csv_Visualization.generate_analysis_report()