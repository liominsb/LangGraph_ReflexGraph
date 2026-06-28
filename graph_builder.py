from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from graph.state import State
from graph.nodes import chatbot, reflect_llm, task_llm, memory_manager, tool_node

# ── 构建状态图 ──
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
    {"tools": "tool_node", "__end__": END},
)
graph_builder.add_edge("tool_node", "memory_manager")
graph_builder.add_edge("memory_manager", "reflect_llm")
graph_builder.add_edge("reflect_llm", "chatbot")
