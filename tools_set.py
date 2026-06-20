import os
import my_tools
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from graph import my_prompt, utils
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()
EMBED_API_KEY = os.getenv("EMBED_API_KEY")
apiKey = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")


sub_tools = my_tools.get_all_tools(my_tools, "write_to_file", "append_to_file", "ask_human","search_and_replace_code")
# 2. 组装全局注册表供 Prompt 动态读取
TOOL_REGISTRY = {tool.name: tool for tool in sub_tools}
AVAILABLE_TOOLS_DESC = ", ".join(TOOL_REGISTRY.keys())





# 3. 定义高级调度工具
@tool(
    description=f"动态调度子Agent执行任务。必须传入任务指令 query，以及该任务所需的工具列表 tool_names。当前可选的工具为：[{AVAILABLE_TOOLS_DESC}]。")
async def create_sub_agent(query: str, tool_names: list[str]) -> str:
    """动态组装并调度子 Agent"""

    # 过滤出真实存在的工具对象
    selected_tools = [TOOL_REGISTRY[name] for name in tool_names if name in TOOL_REGISTRY]

    if not selected_tools:
        return "任务失败：未选择任何有效工具，或传入了不存在的工具名。"

    print(f"\n⚙️ [动态装配] 为子任务 [{query[:10]}...] 挂载了 {len(selected_tools)} 个工具: {tool_names}")

    llm = init_chat_model("mimo-v2.5",
                          model_provider="openai",
                          api_key=apiKey,
                          base_url=base_url,
                          )

    # 动态编译子 Agent
    sub_agent_graph = create_agent(
        model=llm,
        tools=selected_tools,
        system_prompt=my_prompt.sub_prompt
    )

    sub_input = {"messages": [{"role": "user", "content": query}]}
    result = await sub_agent_graph.ainvoke(sub_input)

    last_message = result["messages"][-1]

    # 复用计费逻辑
    utils.parse_and_print_token_usage(last_message, task_name=query, agent_type="子Agent")

    return last_message.content

base_tools = my_tools.get_all_tools(my_tools)
all_tools = base_tools + [create_sub_agent]