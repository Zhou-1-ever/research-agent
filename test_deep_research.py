"""deep_research 的离线测试：mock 掉 LLM 和检索，只验证「广度 × 深度」的递归形状。

不联网、不调模型，结论确定。主要验证三件事：
1. JSON 解析（含模型偶尔包说明文字的兜底）
2. 递归形状：breadth 逐层减半、depth 逐层递减、树按 4×2 展开
3. 最终报告末尾自动附上来源
"""

from unittest.mock import patch

import deep_research
import search_backend


# =========================
# 1. JSON 解析（纯函数）
# =========================

def test_parse_json_direct():
    assert deep_research._parse_json('{"a": 1, "b": [2]}') == {"a": 1, "b": [2]}


def test_parse_json_wrapped_in_prose():
    """模型偶尔把 JSON 包在说明文字里，要能兜底抠出来。"""
    text = '好的，结果如下：\n{"queries": [{"query": "x"}]}'
    assert deep_research._parse_json(text) == {"queries": [{"query": "x"}]}


# =========================
# 2. 递归形状：breadth 减半、depth 递减
# =========================

# 每次调用都记下来，供断言统计
GEN_WIDTHS = []       # generate_serp_queries 收到的 num_queries
PROC_FOLLOWUPS = []   # process_search_result 收到的 num_follow_up_questions


def _fake_generate(query, num_queries, learnings=None):
    GEN_WIDTHS.append(num_queries)
    return [{"query": f"q-{i}", "research_goal": f"goal-{i}"} for i in range(num_queries)]


def _fake_process(query, result_text, num_learnings=3, num_follow_up_questions=3):
    PROC_FOLLOWUPS.append(num_follow_up_questions)
    return {
        "learnings": [f"L:{query}"],
        "follow_up_questions": [f"F:{query}"],
    }


def _fake_search_items(query, top_k=3):
    return [{"title": f"t:{query}", "content": "c", "url": f"https://x/{query}"}]


def test_deep_research_halves_breadth_each_level():
    """breadth=4, depth=2 → 顶层 4 个方向，每支再拆 ceil(4/2)=2 个，共 5 次拆方向。"""
    GEN_WIDTHS.clear()
    PROC_FOLLOWUPS.clear()

    with patch.object(deep_research, "generate_serp_queries", side_effect=_fake_generate), \
         patch.object(deep_research, "process_search_result", side_effect=_fake_process), \
         patch.object(search_backend, "search_items", side_effect=_fake_search_items):
        result = deep_research.deep_research("topic", breadth=4, depth=2)

    # 拆方向：顶层 1 次 breadth=4，第二层 4 支各 breadth=2
    assert GEN_WIDTHS.count(4) == 1, GEN_WIDTHS
    assert GEN_WIDTHS.count(2) == 4, GEN_WIDTHS
    assert len(GEN_WIDTHS) == 5, GEN_WIDTHS

    # 提炼：第一层 4 次（后续问题数=2），第二层 4×2=8 次（后续问题数=1）
    assert PROC_FOLLOWUPS.count(2) == 4, PROC_FOLLOWUPS
    assert PROC_FOLLOWUPS.count(1) == 8, PROC_FOLLOWUPS
    assert len(PROC_FOLLOWUPS) == 12, PROC_FOLLOWUPS

    # 汇总后 learnings / URLs 都去重了
    assert len(result["learnings"]) == len(set(result["learnings"]))
    assert len(result["visited_urls"]) == len(set(result["visited_urls"]))
    assert result["visited_urls"], "应收集到来源 URL"


def test_deep_research_depth_one_does_not_recurse():
    """depth=1 → 只搜一层就触底，不递归。"""
    GEN_WIDTHS.clear()
    PROC_FOLLOWUPS.clear()

    with patch.object(deep_research, "generate_serp_queries", side_effect=_fake_generate), \
         patch.object(deep_research, "process_search_result", side_effect=_fake_process), \
         patch.object(search_backend, "search_items", side_effect=_fake_search_items):
        result = deep_research.deep_research("topic", breadth=3, depth=1)

    assert GEN_WIDTHS == [3], GEN_WIDTHS
    assert len(PROC_FOLLOWUPS) == 3, PROC_FOLLOWUPS
    # depth=1 时每支的后续问题数 = ceil(3/2) = 2，但不会再往下钻
    assert PROC_FOLLOWUPS == [2, 2, 2], PROC_FOLLOWUPS
    assert len(result["learnings"]) == 3


# =========================
# 3. 报告末尾附来源
# =========================

def test_write_final_report_appends_sources():
    with patch.object(deep_research, "_call_llm_text", return_value="REPORT BODY"):
        report = deep_research.write_final_report(
            "主题", ["L1", "L2"], ["https://a", "https://b"]
        )

    assert report.startswith("REPORT BODY"), report
    assert "## Sources" in report, report
    assert "https://a" in report and "https://b" in report, report


# =========================
# 4. 运行入口
# =========================

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}\n      {exc}")
        except Exception as exc:
            failed += 1
            print(f"ERROR {fn.__name__}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {fn.__name__}")

    passed = len(tests) - failed
    print(f"\n{passed}/{len(tests)} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
