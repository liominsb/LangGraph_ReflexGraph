import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class State(TypedDict):
    messages: Annotated[list, add_messages]
    # 反思记忆（覆盖模式：str）
    reflection: str
    # 任务规划（覆盖模式：str，由 task_llm 节点生成）
    task_plan: str
    # 任务进度（覆盖模式：str，由 reflect_llm 节点更新）
    task_progress: str
    total_tokens: Annotated[int, operator.add]


class ReflectionOutput(BaseModel):
    """反思节点的结构化输出"""
    diagnosis: str = Field(description="诊断结论：如果一切正常填'一切正常'，否则填纠错指令")
    progress: str = Field(default="", description="执行进度：各步骤的完成状态")
    current_status: str = Field(default="", description="当前状态：一句话描述当前在做什么")
