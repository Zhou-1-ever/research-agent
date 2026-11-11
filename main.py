from agent import run
from deep_research import deep_research, write_final_report


def _ask_int(prompt_text, default):
    """读一个整数，空输入或非法输入回退默认值。"""
    raw = input(prompt_text).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def research_mode():
    """广度 × 深度的深度研究模式：拆方向 → 递归下钻 → 写报告。"""
    topic = input("研究主题？> ").strip()
    if not topic:
        print("主题为空，已取消。")
        return

    breadth = _ask_int("广度 breadth（推荐 2-4，默认 2，越大覆盖角度越多）> ", 2)
    depth = _ask_int("深度 depth（推荐 1-3，默认 1，越大挖得越深）> ", 1)

    print(f"\n开始深度研究：breadth={breadth} depth={depth}\n")
    result = deep_research(topic, breadth=breadth, depth=depth)

    learnings = result["learnings"]
    urls = result["visited_urls"]
    print(f"\n共收集 {len(learnings)} 条要点、{len(urls)} 个来源，开始写报告...\n")

    report = write_final_report(topic, learnings, urls)
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("=" * 40 + " 最终报告 " + "=" * 40)
    print(report)
    print("\n报告已保存到 report.md")


def main():
    print("研究助理已启动。输入你想调研的问题，我会检索并给出带来源的结论。")
    print("输入 research / 深度研究 进入广度×深度深度研究模式。")
    print("输入 quit / exit / q（或 Ctrl+C）退出。\n")

    while True:
        try:
            question = input("你想调研什么？> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q", "退出"):
            print("再见！")
            break
        if question.lower() in ("research", "/research", "深度研究"):
            research_mode()
            print()
            continue

        print()
        print(run(question))
        print()


if __name__ == "__main__":
    main()
