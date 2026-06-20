
import inspect
import json
import os
import py_compile
from datetime import datetime
import re
import chardet
import numexpr  # 推荐使用 numexpr，比直接用 eval 安全得多
import requests
from dotenv import load_dotenv

from langchain_core.tools import tool, BaseTool
from langchain_tavily import TavilySearch
from langgraph.types import interrupt

load_dotenv()
tavily_api_key=os.getenv("TAVILY_API_KEY")

def get_all_tools(module, *exclude):
    """
    扫描模块中的所有工具，并排除指定名称的工具。

    Args:
        module: 要扫描的模块对象。
        *exclude: 要排除的工具名称，可以传入任意多个字符串。
    """
    exclude_names = set(exclude)  # 将传入的名称转为集合，方便快速查找
    all_tools = []
    for name, obj in inspect.getmembers(module):
        if isinstance(obj, BaseTool):
            if obj.name not in exclude_names:
                all_tools.append(obj)
    return all_tools


def get_encoding(file):
    if not os.path.exists(file):
        return 'utf-8'
    with open(file, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']

@tool(description="使用TavilySearch工具进行搜索,当你想要搜索日期相关的请先使用get_today_date工具 传入str关键词，返回搜索结果的字符串信息")
def search_web(query: str) -> str:
    """在互联网上搜索信息（模拟）。实际可调用 Tavily API。"""
    response=TavilySearch(max_results=1,tavily_api_key=tavily_api_key).invoke(query)
    return f"关于 '{query}' 的模拟搜索结果：{response}"

@tool(description="发送HTTP GET请求获取网页内容。传入URL地址，返回网页的文本内容。支持自动编码检测。可指定超时时间（秒）。")
def fetch_url(url: str, timeout: int = 10) -> str:
    """获取指定URL的网页内容"""
    try:
        # 发送GET请求
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # 检查HTTP错误状态码
        
        # 检测编码：优先使用响应头中的编码，否则使用chardet检测
        if response.encoding:
            encoding = response.encoding
        else:
            # 使用chardet检测内容编码
            result = chardet.detect(response.content)
            encoding = result['encoding'] or 'utf-8'
        
        # 使用检测到的编码解码内容
        content = response.content.decode(encoding)
        
        # 返回内容预览（避免过长内容）
        content_length = len(content)
        if content_length > 2000:
            preview = content[:2000] + f"\n\n... [内容已截断，总长度：{content_length} 字符]"
        else:
            preview = content
            
        return f"成功获取URL '{url}' 的内容（长度：{content_length} 字符）：\n{preview}"
        
    except requests.exceptions.Timeout:
        return f"请求超时：URL '{url}' 在 {timeout} 秒内未响应"
    except requests.exceptions.HTTPError as e:
        return f"HTTP错误：{e}"
    except requests.exceptions.ConnectionError:
        return f"连接错误：无法连接到 '{url}'，请检查网络或URL是否正确"
    except requests.exceptions.RequestException as e:
        return f"请求失败：{str(e)}"
    except UnicodeDecodeError:
        return f"编码错误：无法解码URL '{url}' 的内容，请尝试指定编码"
    except Exception as e:
        return f"获取URL内容失败：{str(e)}"

@tool(description="查询今天的日期，无传入，返回字符串信息")
def get_today_date() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"今天的日期是：{today}"


@tool(description="【信息采集/聊天专用】当你发现当前上下文中缺少用户的关键信息（如不知道用户的名字、偏好等），需要主动向用户发起提问时调用。参数传入你具体的提问语句，返回用户的真实回答。")
def ask_human(question: str) -> str:
    """向人类提问并获取回答"""
    # 挂起并向外部展示 AI 的问题
    human_response = interrupt({
        "ai_wants_to_ask": question
    })

    # 假设你的恢复指令是 Command(resume={"data": "你的回答"})
    return human_response.get("data", "人类未提供有效回答")

# @tool(description="【档案审批与状态入库专用】当且仅当你已经明确收集到了用户的名字 (name) 和昵称 (nickname) 时调用！将提取到的数据提交给人类进行核对并覆盖到系统全局状态中。严禁在尚未知道具体信息时，将此工具当作提问工具使用。")
# def human_assistance(name: str, nickname: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
#     """请求人类协助。"""
#     human_response = interrupt(
#         {
#             "question": "这是正确的吗?",
#             "name": name,
#             "nickname": nickname,
#         },
#     )
#     if human_response.get("correct", "").lower().startswith("y"):
#         verified_name = name
#         verified_nickname = nickname
#         response = "Correct"
#     else:
#         verified_name = human_response.get("name", name)
#         verified_nickname = human_response.get("nickname", nickname)
#         response = f"Made a correction: {human_response}"
#
#     state_update = {
#         "name": verified_name,
#         "nickname": verified_nickname,
#         "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
#     }
#     return Command(update=state_update)


@tool(description="执行精确的数学计算。传入数学表达式字符串（例如 '2.5 * (100 - 34)'），返回计算结果。")
def calculate(expression: str) -> str:
    """安全的数学计算工具，弥补大模型算力短板。"""
    try:
        # numexpr.evaluate 支持大部分标准数学运算
        result = str(numexpr.evaluate(expression))
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算失败，请检查表达式格式。错误信息：{str(e)}"

@tool(description="将文本内容保存到本地文件。传入文件路径和内容，返回操作结果。")
def write_to_file(filepath: str, content: str) -> str:
    """让 Agent 可以固化输出结果（如生成报告、写代码）。"""
    encoding=get_encoding(filepath)
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)
        return f"成功！内容已写入文件：{os.path.abspath(filepath)}"
    except Exception as e:
        return f"写入文件失败：{str(e)}"


@tool(description="读取指定目录的文件列表。传入目标路径（默认为当前目录 '.'），返回包含所有文件和子目录名称的字符串。")
def get_listdir(path: str = ".") -> str:
    """带路径遍历的文件导航工具"""
    try:
        if not os.path.exists(path):
            return f"错误：路径不存在 - {path}"
        if not os.path.isdir(path):
            return f"错误：目标不是一个目录 - {path}"

        items = os.listdir(path)
        if not items:
            return f"目录 '{path}' 为空。"

        # 区分文件和文件夹，方便 Agent 理解结构
        dirs = [d for d in items if os.path.isdir(os.path.join(path, d))]
        files = [f for f in items if os.path.isfile(os.path.join(path, f))]

        formatted_list = f"目录 [{os.path.abspath(path)}] 的内容：\n"
        if dirs:
            formatted_list += "📁 文件夹:\n" + "\n".join(f"  - {d}/" for d in dirs) + "\n"
        if files:
            formatted_list += "📄 文件:\n" + "\n".join(f"  - {f}" for f in files)

        return formatted_list

    except Exception as e:
        return f"读取目录失败，错误信息：{str(e)}"

@tool(description="读取指定路径的文件的内容。默认带有行号以便于后续的精确修改。如果文件很大，建议先读取关键部分。")
def read_from_file(filepath: str) -> str:
    """读取文件并自动追加行号"""
    encoding = get_encoding(filepath)
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            lines = f.readlines()

        # 限制单次读取的大小，防止撑爆 Context
        if len(lines) > 300:
            return f"错误：文件过大（{len(lines)}行）。为了安全，请使用专门的局部读取工具。"

        numbered_content = ""
        for i, line in enumerate(lines, start=1):
            # 将每一行格式化为 "1: import os"
            numbered_content += f"{i:4d} | {line}"

        return f"文件 {filepath} 的内容如下：\n{numbered_content}"
    except Exception as e:
        return f"错误：无法读取文件 - {e}"


@tool(description="【代码检索工具】在文件中搜索指定的关键字或正则表达式，返回匹配行的上下文（含行号）。用于在修改代码前定位目标函数或变量。")
def grep_search(filepath: str, pattern: str, context_lines: int = 2) -> str:
    """正则表达式全局检索工具"""
    encoding = get_encoding(filepath)
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            lines = f.readlines()

        results = []
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                # 提取匹配行的前后文
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)

                results.append(f"--- 匹配结果 (行 {i + 1}) ---")
                for j in range(start, end):
                    prefix = ">>" if j == i else "  "
                    results.append(f"{j + 1:4d} {prefix} {lines[j].rstrip()}")

        if not results:
            return f"未在 {filepath} 中找到匹配 '{pattern}' 的内容。"

        return "\n".join(results)
    except Exception as e:
        return f"检索失败：{str(e)}"


@tool(description="【修改代码核心工具】在文件中搜索一段精确的旧代码块，并将其替换为新代码块。必须保证 old_block 与文件中的原文（包括空格和缩进）完全一致。")
def search_and_replace_code(filepath: str, old_block: str, new_block: str) -> str:
    """基于精确代码块匹配的替换工具，防范大模型行号幻觉。"""
    encoding = get_encoding(filepath)
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            content = f.read()

        # 检查旧代码块是否存在，且只存在一处（防止误替换）
        count = content.count(old_block)
        if count == 0:
            return f"替换失败：在文件 {filepath} 中未找到完全匹配的 old_block。请检查缩进、换行或特殊字符是否绝对一致。"
        if count > 1:
            return f"替换失败：在文件 {filepath} 中找到 {count} 处匹配的 old_block。目标必须是唯一的，请包含更多的上下文代码以确保 old_block 的唯一性。"

        # 执行替换
        new_content = content.replace(old_block, new_block)

        with open(filepath, 'w', encoding=encoding) as f:
            f.write(new_content)

        return f"代码修改成功！已在 {filepath} 中完成指定代码块的替换。"
    except Exception as e:
        return f"代码替换发生系统错误：{str(e)}"

@tool(description="【追加内容必用】在文件的最后追加内容，传入文件路径和内容，返回操作结果。")
def append_to_file(filepath: str, content: str) -> str:
    """在文件的最后追加内容，传入文件路径和内容，返回操作结果。"""
    encoding=get_encoding(filepath)
    try:
        with open(filepath, 'a', encoding=encoding) as f:
            f.write(content)
        return f"成功！内容已追加到文件：{os.path.abspath(filepath)}"
    except Exception as e:
        return f"追加文件失败：{str(e)}"

@tool(description="调用外部API并处理响应，支持认证和错误处理")
def call_api(url: str, method: str = "GET", headers: dict = None, data: dict = None, json_data: dict = None, auth: dict = None, timeout: int = 10) -> str:
    """
    通用的API调用工具，支持各种HTTP方法和认证方式。
    
    参数:
        url: API端点URL
        method: HTTP方法（GET, POST, PUT, DELETE, PATCH等）
        headers: 请求头字典
        data: 表单数据（用于POST/PUT）
        json_data: JSON数据（用于POST/PUT）
        auth: 认证信息，格式：{"type": "basic/bearer/apikey", "username": "", "password": "", "token": "", "api_key": "", "header_name": ""}
        timeout: 超时时间（秒）
    
    返回:
        包含响应状态、内容和错误信息的字符串
    """
    try:
        # 准备请求头
        request_headers = headers or {}
        
        # 处理认证
        auth_obj = None
        if auth:
            auth_type = auth.get("type", "").lower()
            if auth_type == "basic":
                from requests.auth import HTTPBasicAuth
                auth_obj = HTTPBasicAuth(auth.get("username", ""), auth.get("password", ""))
            elif auth_type == "bearer":
                token = auth.get("token", "")
                request_headers["Authorization"] = f"Bearer {token}"
            elif auth_type == "apikey":
                api_key = auth.get("api_key", "")
                header_name = auth.get("header_name", "X-API-Key")
                request_headers[header_name] = api_key
        
        # 发送请求
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=request_headers,
            data=data,
            json=json_data,
            auth=auth_obj,
            timeout=timeout
        )
        
        # 尝试解析JSON响应
        try:
            response_json = response.json()
            response_content = json.dumps(response_json, indent=2, ensure_ascii=False)
        except:
            response_content = response.text
        
        # 检查响应长度
        content_length = len(response_content)
        if content_length > 3000:
            preview = response_content[:3000] + f"\n\n... [内容已截断，总长度：{content_length} 字符]"
        else:
            preview = response_content
        
        # 构建响应信息
        result = [
            f"API调用成功！",
            f"URL: {url}",
            f"方法: {method.upper()}",
            f"状态码: {response.status_code}",
            f"响应头: {dict(response.headers)}",
            f"响应内容（长度：{content_length} 字符）:\n{preview}"
        ]
        
        return "\n".join(result)
        
    except requests.exceptions.Timeout:
        return f"请求超时：URL '{url}' 在 {timeout} 秒内未响应"
    except requests.exceptions.HTTPError as e:
        return f"HTTP错误：{e}"
    except requests.exceptions.ConnectionError:
        return f"连接错误：无法连接到 '{url}'，请检查网络或URL是否正确"
    except requests.exceptions.RequestException as e:
        return f"请求失败：{str(e)}"
    except Exception as e:
        return f"API调用失败：{str(e)}"

@tool(description="检查 Python 文件的基础语法（不执行代码）。修改 Python 代码后必须调用此工具进行验证。")
def check_python_syntax(filepath: str) -> str:
    """防御代码被修改至崩溃的最后一道防线。"""
    if not filepath.endswith('.py'):
        return "这不是一个 Python 文件，跳过语法检查。"
    try:
        py_compile.compile(filepath, doraise=True)
        return "语法检查通过：未发现基础语法错误（SyntaxError）。"
    except py_compile.PyCompileError as e:
        return f"严重错误！代码修改导致了语法崩溃：\n{str(e)}"
