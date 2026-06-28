from typing import Annotated
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool, InjectedToolCallId
from langchain_mcp_adapters.client import MultiServerMCPClient

from graph import config, my_tools, my_prompt, utils


sub_tools = my_tools.get_all_tools(my_tools, "write_to_file", "append_to_file", "ask_human","search_and_replace_code")
# 2. 组装全局注册表供 Prompt 动态读取
TOOL_REGISTRY = {tool.name: tool for tool in sub_tools}
AVAILABLE_TOOLS_DESC = ", ".join(TOOL_REGISTRY.keys())





# 3. 定义高级调度工具
@tool(
    description=f"动态调度子Agent执行任务。必须传入任务指令 query与needs_expert,tool_call_id，以及该任务所需的工具列表 tool_names。当前可选的工具为：[{AVAILABLE_TOOLS_DESC}]。"
    "如果任务仅为搜索、读取等机械动作，needs_expert 设为 False；如果任务涉及代码 Bug 修复、逻辑重构并需要生成补丁，needs_expert 必须设为 True。"
)
async def create_sub_agent(query: str, tool_names: list[str],tool_call_id: Annotated[str, InjectedToolCallId], needs_expert: bool = False) -> Command|str:
    """动态组装并调度子 Agent"""
    # 过滤出真实存在的工具对象
    selected_tools = [TOOL_REGISTRY[name] for name in tool_names if name in TOOL_REGISTRY]

    if not selected_tools:
        return "任务失败：未选择任何有效工具，或传入了不存在的工具名。"

    print(f"\n⚙️ [动态装配] 为子任务 [{query[:10]}...] 挂载了 {len(selected_tools)} 个工具: {tool_names}")

    if needs_expert:
        model_name = "mimo-v2.5-pro"
    else:
        model_name = "mimo-v2.5"

    llm = init_chat_model(model=model_name,
                          model_provider="openai",
                          api_key=config.LLM_API_KEY,
                          base_url=config.LLM_BASE_URL,
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

    current_cost=utils.get_total_tokens(last_message)

    return Command(
        update={
            # 触发操作符自动累加 Token
            "total_tokens": current_cost,
            # 手动追加工具执行结果给主 Agent
            "messages": [
                ToolMessage(
                    content=last_message.content,
                    tool_call_id=tool_call_id,
                    name="create_sub_agent"
                )
            ]
        }
    )

@tool(description="【涉及github专用】使用集成github工具包的Agent执行任务，只有读取权限。必须传入任务指令 query和tool_call_id")
async def create_github_sub_agent(query: str,tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    client = MultiServerMCPClient({
        "github": {
            "command": "github-mcp-server.exe",
            "args": ["stdio", "--read-only"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": config.GITHUB_API_KEY},
            "transport": "stdio",
        }
    })
    llm = init_chat_model("mimo-v2.5",
                          model_provider="openai",
                          api_key=config.LLM_API_KEY,
                          base_url=config.LLM_BASE_URL,
                          )

    # 动态编译子 Agent
    sub_agent = create_agent(
        model=llm,
        tools=await client.get_tools(),
        system_prompt=my_prompt.sub_prompt
    )
    sub_input = {"messages": [{"role": "user", "content": query}]}
    result = await sub_agent.ainvoke(sub_input)
    last_message = result["messages"][-1]
    current_cost=utils.get_total_tokens(last_message)

    return Command(
        update={
            # 触发操作符自动累加 Token
            "total_tokens": current_cost,
            # 手动追加工具执行结果给主 Agent
            "messages": [
                ToolMessage(
                    content=last_message.content,
                    tool_call_id=tool_call_id,
                    name="create_github_sub_agent"
                )
            ]
        }
    )



base_tools = my_tools.get_all_tools(my_tools)
all_tools = base_tools + [create_sub_agent]+[create_github_sub_agent]