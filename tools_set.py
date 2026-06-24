import os
from typing import re

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

import my_tools
from graph import my_prompt, utils

load_dotenv()
EMBED_API_KEY = os.getenv("EMBED_API_KEY")
apiKey = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")
github_key = os.getenv("GITHUB_API_KEY")


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

@tool(description="【涉及github专用】使用集成github工具包的Agent执行任务，只有读取权限。必须传入任务指令 query")
async def create_github_sub_agent(query: str) -> str:
    client = MultiServerMCPClient({
        "github": {
            "command": "github-mcp-server.exe",
            "args": ["stdio", "--read-only"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": github_key},
            "transport": "stdio",
        }
    })
    llm = init_chat_model("mimo-v2.5",
                          model_provider="openai",
                          api_key=apiKey,
                          base_url=base_url,
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
    utils.parse_and_print_token_usage(last_message, task_name=query, agent_type="子Agent")

    return last_message.content

@tool(description="【Prompt测试工具】用指定的 system_prompt 启动子Agent，执行一个具体的测试任务，然后自动评分。用于验证 prompt 的效果。传入 task（一个具体的测试任务，如'读取main.py第50行并定位Bug'）、custom_prompt（要测试的系统提示词）、tool_names（子Agent可用的工具列表）。返回子Agent的实际表现和评分。")
async def evolve_agent(task: str, custom_prompt: str, tool_names: list[str], score_criteria: str = "") -> str:
    """一次完整的迭代进化循环：执行 → 评分 → 记录"""

    # ========== 第一步：启动子 Agent ==========
    selected_tools = [TOOL_REGISTRY[name] for name in tool_names if name in TOOL_REGISTRY]
    if not selected_tools:
        return "错误：未选择任何有效工具。可用工具：" + AVAILABLE_TOOLS_DESC

    print(f"\n🧬 [进化迭代] 任务: {task[:30]}...")

    sub_llm = init_chat_model("mimo-v2.5",
                              model_provider="openai",
                              api_key=apiKey,
                              base_url=base_url)

    sub_agent = create_agent(
        model=sub_llm,
        tools=selected_tools,
        system_prompt=custom_prompt
    )

    sub_input = {"messages": [{"role": "user", "content": task}]}
    result = await sub_agent.ainvoke(sub_input)
    agent_output = result["messages"][-1].content
    utils.parse_and_print_token_usage(result["messages"][-1], task_name=task[:20], agent_type="迭代Agent")

    # ========== 第二步：自动评分 ==========
    if not score_criteria:
        score_criteria = f"1. 是否完成了任务目标：{task}\n2. 输出质量（正确性、完整性、可读性）\n3. 有无明显错误或遗漏"

    judge_prompt = f"""你是一个严格的质量评审员。对以下内容按评分标准打分。

【评分标准】：
{score_criteria}

【待评估内容】：
{agent_output[:3000]}

【输出格式（必须严格遵循）】：
总分：X/10
- 维度1：X/10 — 一句话说明
- 维度2：X/10 — 一句话说明
- 维度3：X/10 — 一句话说明
改进建议：（列出必须修复的 top 3 问题）
是否通过（≥10分通过）：是/否"""

    score_result = sub_llm.invoke([SystemMessage(content=judge_prompt)])
    score_text = score_result.content

    # ========== 第三步：写入迭代日志 ==========
    log_path = "aa/evolution_log.md"
    os.makedirs("aa", exist_ok=True)

    # 如果文件不存在，写入表头
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# 🧬 迭代进化日志\n\n")

    # 追加本轮记录
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"---\n\n")
        f.write(f"### 任务\n{task}\n\n")
        f.write(f"### 使用的 Prompt\n```\n{custom_prompt}\n```\n\n")
        f.write(f"### 子Agent输出\n```\n{agent_output[:2000]}\n```\n\n")
        f.write(f"### 评分结果\n{score_text}\n\n")

    # ========== 第四步：返回结果给主 Agent ==========
    print(f"📊 [进化评分]\n{score_text}")

    score_match = re.search(r'总分[：:]\s*(\d+)', score_text)
    score_num = score_match.group(1) if score_match else "?"

    # 只提取"改进建议"部分（3行以内）
    improvements = ""
    if "改进建议" in score_text:
        imp_section = score_text.split("改进建议")[1].split("是否通过")[0]
        improvements = imp_section.strip()[:200]

    # 精简返回，不返回完整输出
    return (
        f"【迭代结果】任务: {task[:50]}...\n"
        f"总分: {score_num}/10\n"
        f"改进建议: {improvements}\n"
        f"📁 完整记录已写入 aa/evolution_log.md"
    )

base_tools = my_tools.get_all_tools(my_tools)
all_tools = base_tools + [create_sub_agent]+[create_github_sub_agent]+[evolve_agent]