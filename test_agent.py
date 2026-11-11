
import os

import agent
import search_backend
import tools


# =========================
# 配置自检
# =========================

REQUIRED_ENV = ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL", "TAVILY_API_KEY")


def missing_config():
    """返回 .env 里缺失的配置项名。"""
    return [name for name in REQUIRED_ENV if not os.getenv(name)]


def _normalize(text):
    """压掉千分位和空白，免得「56,088」这类格式差异干扰断言。"""
    return text.replace(",", "").replace("，", "").replace(" ", "")


# =========================
# 1. 工具层：不联网，结论确定
# =========================

def test_execute_tool_returns_product():
    assert tools.execute_tool("calculator", {"a": 123, "b": 456}) == "56088"
    assert tools.execute_tool("calculator", {"a": 12, "b": 12}) == "144"


def test_unknown_tool_is_reported_as_string():
    result = tools.execute_tool("nonexistent", {})
    assert result == "Error: 未知工具 'nonexistent'", result


def test_missing_and_wrong_arg_names_become_error_strings():
    """参数写错不该炸循环 —— execute_tool 这一层要统一兜住。"""
    assert tools.execute_tool("calculator", {"a": 1}).startswith("Error: TypeError")
    assert tools.execute_tool("calculator", {"x": 1, "y": 2}).startswith("Error: TypeError")


# ---- 注册表 ----

def test_registry_covers_all_tools():
    """三个工具都得出现在发给模型的 schema 里，且描述和必填项齐全。"""
    schemas = tools.tool_schemas()
    names = {item["function"]["name"] for item in schemas}
    assert names == {"calculator", "search", "fetch"}, names

    for item in schemas:
        assert item["type"] == "function"
        fn = item["function"]
        assert fn["description"], fn["name"]
        assert fn["parameters"]["required"], fn["name"]


# =========================
# 2. 检索排版层：纯函数，不联网，结论确定
# =========================
#
# search/fetch 的真实联网逻辑依赖 Tavily key 和网络，不可离线测；
# 但把「排版」抽成纯函数后，这部分仍能确定性地测。

FAKE_RESULTS = [
    {"title": "文章一", "content": "摘要一", "url": "https://example.com/1"},
    {"title": "文章二", "content": "摘要二", "url": "https://example.com/2"},
]


def test_format_search_results_formats_candidates():
    out = search_backend.format_search_results("测试", FAKE_RESULTS)
    assert "搜索结果" in out, out
    assert "共 2 条" in out, out
    assert "文章一" in out, out
    assert "URL: https://example.com/1" in out, out


def test_format_search_results_no_match_is_not_an_error():
    """搜不到是「没找到」，不是「调用出错」—— 要让模型换关键词而不是纠错重试。"""
    out = search_backend.format_search_results("量子计算 育种", [])
    assert "未找到" in out, out
    assert "Error" not in out, out


def test_format_fetched_content_returns_body():
    out = search_backend.format_fetched_content("https://example.com/1", "这是正文内容")
    assert "正文" in out, out
    assert "这是正文内容" in out, out


def test_format_fetched_content_truncates_long_body():
    """长正文必须被截断，否则一次 fetch 就能把上下文撑爆。"""
    long_body = "字" * (search_backend.MAX_FETCH_CHARS + 100)
    out = search_backend.format_fetched_content("https://example.com/1", long_body)

    assert "正文已截断" in out, out[-80:]
    # 截断标记里报告了原文总字数，它必须大于上限，否则根本不该发生截断
    total = int(out.split("共 ")[1].split(" 字")[0])
    assert total > search_backend.MAX_FETCH_CHARS, total
    # 实际交付给模型的文本必须比原文短，才说明真的切了
    assert len(out) < total, (len(out), total)


# =========================
# 3. 端到端：真实调用，验证 Loop 跑完整
# =========================

def test_single_calculation():
    """单次工具调用闭环：提问 → tool_calls → 执行 → 最终回答。"""
    answer = agent.run("请用 calculator 工具计算 123 × 456，然后只告诉我结果。")
    assert "56088" in _normalize(answer), answer


def test_chained_calculations():
    """连续两轮工具调用，顺带验证 assistant 消息和结果被正确回填进上下文。"""
    answer = agent.run(
        "分两步做：先用 calculator 算出 12 乘以 12，"
        "再用 calculator 把上一步的结果乘以 2。最后告诉我最终结果。"
    )
    assert "288" in _normalize(answer), answer


def test_plain_answer_without_tool():
    """不需要工具时，第一轮就该返回，不该空转。"""
    answer = agent.run("只回两个字：你好。不要调用任何工具。")
    assert answer and answer.strip(), "应返回非空回答"
    assert "Error" not in answer, answer


# ---- research 场景 ----

def test_research_answer_cites_url():
    """结论必须带来源，这是 system prompt 立的规矩。"""
    answer = agent.run("检索 asyncio 和多线程怎么选，给出结论并附上来源链接。")
    assert "http" in answer, answer


def test_mixed_tools_in_one_loop():
    """一个 Loop 同时挂三个异质工具，验证分发没串。"""
    answer = agent.run(
        "做三件事：1) 用 calculator 算 137 乘以 29；"
        "2) 用 search 检索 asyncio 的适用场景；"
        "3) 把计算结果和检索到的来源链接一起告诉我。"
    )
    normalized = _normalize(answer)
    assert "3973" in normalized, answer
    assert "http" in answer, answer


# =========================
# 4. 运行入口
# =========================

if __name__ == "__main__":
    missing = missing_config()
    if missing:
        print("缺少配置：" + ", ".join(missing))
        print("请先在 .env 里补全，再重跑。")
        raise SystemExit(2)

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    errored = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}\n      {exc}")
        except Exception as exc:
            # 真实调用才会走到这里：网络不通、key 无效、超时等
            errored += 1
            print(f"ERROR {fn.__name__}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {fn.__name__}")

    passed = len(tests) - failed - errored
    print(f"\n{passed}/{len(tests)} passed, {failed} failed, {errored} errored")
    raise SystemExit(1 if (failed or errored) else 0)
