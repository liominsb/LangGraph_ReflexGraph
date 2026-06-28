# 多智能体编排系统 (LangGraph_ReflexGraph)

基于 LangGraph 构建的多智能体编排系统，具备任务规划、动态子Agent调度和执行反思诊断能力。

## 功能特性

- **主控Agent**：负责任务规划、工具调用决策、结果聚合，拥有唯一的代码修改权限
- **动态子Agent**：根据任务需求自动装配工具，异步执行子任务，采用只读隔离策略
- **反思诊断**：审查执行轨迹，检测工具崩溃、幻觉参数、死循环和目标偏离，诊断结果以最高优先级注入下一轮决策
- **主从PR代码审查**：子Agent侦察定位返回补丁方案，主控节点统一执行代码变更并强制语法校验
- **Token统计**：实时追踪主Agent与子Agent消耗
- **人类交互**：支持中断-恢复的人机协作模式，以及对话状态回溯
- **上下文压缩**：自动管理上下文窗口，超过阈值时触发摘要压缩历史记录

## 工作流程

系统采用多层 Agent 编排架构，核心工作流如下：

1. **任务规划器**前置解析用户意图，输出结构化任务拆解
2. **主控Agent**根据任务复杂度决定直接调用工具或通过 `create_sub_agent` 调度子Agent
3. 工具执行完毕后，**上下文压缩节点**检查Token占用，必要时进行摘要压缩
4. **反思诊断节点**审查最近6条消息的执行轨迹
5. 若检测到异常，以"最高优先级纠错指令"注入系统消息，强制主控Agent中断原计划并修复
6. 若一切正常，继续推进任务直到完成

![Graph](graph.png)

## 内置工具

### 基础工具

| 工具 | 说明 |
|------|------|
| `search_web` | Tavily 网络搜索，传入关键词返回搜索结果 |
| `fetch_url` | 获取网页内容，支持自动编码检测与超时控制 |
| `calculate` | 基于 numexpr 的安全数学计算 |
| `get_today_date` | 获取当前日期 |
| `read_from_file` | 读取文件内容并自动追加行号，限制1000行防撑爆上下文 |
| `write_to_file` | 将内容写入文件 |
| `append_to_file` | 向文件末尾追加内容 |
| `get_listdir` | 列出目录文件，区分文件夹和文件 |
| `ask_human` | 向人类提问，支持 interrupt 挂起与 Command 恢复 |
| `call_api` | 通用API调用，支持 Basic/Bearer/APIKey 三种认证方式 |

### 代码编辑与执行工具

| 工具 | 说明 |
|------|------|
| `grep_search` | 正则表达式代码检索，返回匹配行及前后上下文（含行号） |
| `search_and_replace_code` | 精确代码块替换，强制唯一匹配防误替换 |
| `check_python_syntax` | Python 文件语法校验，代码修改后必须调用验证 |
| `run_python_file` | 运行指定的 Python 文件，返回 stdout、stderr 和退出码 |
| `run_python_snippet` | 直接运行一段 Python 代码片段，返回执行结果 |

### 调度工具

| 工具 | 说明 |
|------|------|
| `create_sub_agent` | 动态调度子Agent，按需装配指定工具集异步执行子任务 |
| `create_github_sub_agent` | 使用集成 GitHub MCP 工具包的 Agent 执行只读任务 |

## 快速开始

### 1. 安装依赖

```bash
pip install langgraph langchain langchain-core langchain-openai langchain-tavily python-dotenv numexpr requests chardet httpx tiktoken langchain_mcp_adapters
```

建议使用 Python 3.10+ 版本。

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
LLM_API_KEY=你的API密钥
LLM_BASE_URL=你的API地址（需兼容OpenAI格式）
TAVILY_API_KEY=你的Tavily密钥（前往 https://app.tavily.com 免费申请）
GITHUB_API_KEY=你的GitHub PAT (用于 MCP server)
```

### 3. 运行

```bash
python main.py
```

启动后直接输入问题即可交互：

```
> 帮我分析一下这个项目的结构
> 计算 (123 + 456) * 789
> 搜索最新的AI发展趋势
```

### 命令说明

| 命令 | 说明 |
|------|------|
| `exit` | 退出程序 |
| `<<1` | 回退到上一步已完成的 checkpoint 状态并恢复执行，用于对话中途纠错 |
| `END` | 结束多行输入，发送消息给 Agent |

## 文件结构与架构拆分

系统采用职责单一的模块化架构：

| 文件 | 说明 |
|------|------|
| `main.py` | 主程序 CLI 交互入口，包含会话管理和状态流处理 |
| `state.py` | 数据模型定义，包含 `State` 和 `ReflectionOutput` |
| `config.py` | 统一配置管理，包含环境变量加载和 LLM 实例初始化 |
| `nodes.py` | 图节点逻辑实现（chatbot、task_llm、reflect_llm、memory_manager） |
| `graph_builder.py` | LangGraph 状态图构建与边连接 |
| `my_tools.py` | 所有底层工具函数定义 |
| `tools_set.py` | 工具集合聚合、子Agent动态调度逻辑 |
| `my_prompt.py` | 全部 Prompt 模板管理 |
| `utils.py` | Token 统计计算与会话计数器存取工具 |
| `counter.json` | 自动维护的本地会话计数器 |
| `checkpoints.db` | SQLite 状态持久化数据库 |

## 系统配置

### Prompt模板

在 `my_prompt.py` 中可修改：

| 变量 | 说明 |
|------|------|
| `one_prompt` | 主控Agent系统提示词，包含任务分流策略、主从PR代码审查规范、规划存档流程 |
| `sub_prompt` | 子Agent系统提示词，定义只读隔离策略与补丁输出格式 |
| `task_prompt` | 任务规划器提示词，定义任务拆解规范 |
| `reflect_prompt` | 反思诊断提示词，定义四类异常诊断标准与输出约束 |
| `get_summary_prompt` | 记忆压缩节点提示词，用于生成历史对话摘要 |

### 工具扩展

在 `my_tools.py` 中添加新工具，使用 `@tool` 装饰器定义即可，工具会通过 `get_all_tools` 自动扫描注册：

```python
from langchain_core.tools import tool

@tool(description="工具描述")
def my_tool(param: str) -> str:
    """工具说明"""
    return "结果"
```

若需在子Agent中排除该工具，修改 `tools_set.py` 中的 `exclude` 列表。

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| Tavily 搜索返回空结果 | 确认 `TAVILY_API_KEY` 已正确配置，前往 https://app.tavily.com 申请免费额度 |
| LLM 调用超时 | 检查 `LLM_BASE_URL` 是否可达，系统已内置 100s 超时 + 最多10次自动重试 |
| `graph.png` 生成失败 | 可视化可能需要额外的依赖库，抛出异常不会影响核心程序运行 |
| `<<1` 回退后无反应 | 确保在此之前至少有一轮完整对话，回退功能依赖已有的 checkpoint 历史 |
