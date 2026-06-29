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
    # 任务步骤列表（覆盖模式：list，由 task_llm 初始化，Agent 通过工具更新状态）
    task_steps: list
    total_tokens: Annotated[int, operator.add]


class ReflectionOutput(BaseModel):
    """反思节点的结构化输出"""
    diagnosis: str = Field(description="诊断结论：如果一切正常填'一切正常'，否则填纠错指令")


class TaskPlanOutput(BaseModel):
    """任务规划节点的结构化输出"""
    plan: str = Field(description="任务总规划：包含任务类型、任务概述、约束与注意事项（markdown格式）")
    steps: list[dict] = Field(description="任务步骤列表，每个元素为 {step: int, title: str}")

    @classmethod
    def _normalize_plan(cls, obj: dict) -> dict:
        """兼容 LLM 把 plan 返回为 dict 的情况，转为 markdown 字符串"""
        if isinstance(obj, dict) and isinstance(obj.get("plan"), dict):
            parts = []
            for k, v in obj["plan"].items():
                parts.append(f"## {k}\n{v}")
            obj["plan"] = "\n\n".join(parts)
        return obj

    @classmethod
    def model_validate(cls, obj, **kwargs):
        if isinstance(obj, dict):
            obj = cls._normalize_plan(obj)
        return super().model_validate(obj, **kwargs)

    @classmethod
    def model_validate_json(cls, json_data, **kwargs):
        import json as _json
        if isinstance(json_data, (str, bytes)):
            obj = _json.loads(json_data)
            obj = cls._normalize_plan(obj)
            json_data = _json.dumps(obj, ensure_ascii=False)
        return super().model_validate_json(json_data, **kwargs)
