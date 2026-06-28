import asyncio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from graph.graph_builder import graph_builder
from graph import utils


def _print_welcome():
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
        session_counter = utils.get_json_counter()
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
                session_counter += 1
                utils.update_json_counter(session_counter)
                config = {"configurable": {"thread_id": f"{session_counter}"}}
                state = await graph.aget_state(config)
                print("\033[2J\033[H", end="", flush=True)
                _print_welcome()
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
                state_config = None
                for a in list_state:
                    if a.next != ():
                        continue
                    else:
                        state_config = a
                        break
                if len(list_state) >= 1:
                    events = graph.astream(None, state_config.config, stream_mode="values")
                    async for event in events:
                        pass  # 即使不需要在终端重复打印，也必须迭代完毕
                    # 必须重新获取跳跃后的最新状态，覆盖主循环的变量！
                    state = await graph.aget_state(config)
                    continue
            if state.interrupts:
                payload = Command(resume={"data": content})
            else:
                payload = {"messages": [{"role": "user", "content": content}]}

            # 统一处理：无论是中断恢复还是普通消息，都需要流式处理
            printed_ids = set()
            async for event in graph.astream(payload, config, stream_mode="values"):
                for msg in event.get("messages", []):
                    msg_id = getattr(msg, "id", None) or hash(msg.content)
                    if msg_id not in printed_ids:
                        msg.pretty_print()
                        printed_ids.add(msg_id)

            # Token 消耗统计移到流式循环外，只打印一次
            state = await graph.aget_state(config)
            total_session_cost = state.values.get("total_tokens", 0)
            print("\n" + "=" * 40)
            print(f"本次交互总计消耗 Token: {total_session_cost}")
            print("=" * 40 + "\n")


if __name__ == "__main__":
    _print_welcome()
    asyncio.run(main())