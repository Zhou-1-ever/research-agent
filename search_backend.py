"""可插拔的检索后端 —— 当前接 Tavily（真实联网检索）。

对外只暴露两个函数：

    search(query, top_k=3) -> str    给 LLM 看的候选列表（标题 / 摘要 / URL）
    fetch(url)             -> str    给 LLM 看的正文

整个项目里**只有这个文件知道「数据从哪来」**。第一版是 stub（硬编码 3 篇
asyncio 语料、不联网），现在换成 Tavily。以后要再换 Serper / Bing 之类，
只改这个文件，tools.py 和 agent.py 一行都不用动 —— 它们自始至终只看到
「传字符串进去、拿字符串出来」。

为了保持可单测，这里刻意把「联网」和「排版」拆成两层：

    - search / fetch 负责联网拿数据（薄薄一层，需要 key 和网络）
    - format_* 是纯函数，把返回的 dict 排版成给模型看的字符串（可离线单测）

这样即便没有 key、没有网络，排版逻辑仍然能被单元测试覆盖。
"""

import os
import time

from dotenv import load_dotenv
from requests.exceptions import RequestException
from tavily import TavilyClient


# 自己读 .env：search_backend 是独立模块，不该依赖别的文件先 load_dotenv。
load_dotenv()


# 正文超过这个长度就截断。真实网页动辄上万字，不截断一次 fetch 就能撑爆上下文。
# 截断是 fetch 这个工具的「输出契约」，所以放在这一层，而不是塞进 Agent Loop。
MAX_FETCH_CHARS = 4000

# 网络重试：代理（Clash）偶发断连时会抛 SSL EOF 这类瞬时错误，
# 重试几次而不是直接失败。认证错误（key 无效）不在这类里，不会重试。
# 次数与间隔可在 .env 里配置。
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
RETRY_DELAY_SECONDS = float(os.getenv("RETRY_DELAY_SECONDS", "1.0"))


def _client() -> TavilyClient:
    """惰性创建 Tavily 客户端。

    不放在模块顶层 import 时创建：导入不该因为没有 key 就失败，只有真正
    调用 search/fetch 时才需要 key。缺 key 抛出的异常会被 execute_tool
    统一兜住、转成字符串回传给模型。
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("未配置 TAVILY_API_KEY，请先在 .env 里补全")
    return TavilyClient(api_key=api_key)


def _with_retry(fn):
    """对瞬时网络错误重试，其余异常直接抛。

    只重试 RequestException（连接失败 / 超时 / SSL 断连），认证失败、参数错误
    这类重试也没用的问题不会进来。最终仍失败就抛出，交给 execute_tool 兜住。
    """
    last_exc = None
    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            return fn()
        except RequestException as exc:
            last_exc = exc
            if attempt < RETRY_MAX_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY_SECONDS)
    raise last_exc


# =========================
# 排版层（纯函数，可离线单测）
# =========================


def format_search_results(query, results) -> str:
    """把 Tavily search 返回的 results 排版成候选列表。

    搜不到时返回一句说明而不是 Error —— 这是「没找到」，不是「调用出错」，
    要让模型去换关键词，而不是去纠错重试。
    """
    if not results:
        return f"未找到与 {query!r} 相关的结果。"

    lines = [f"搜索结果（共 {len(results)} 条）："]
    for index, item in enumerate(results, 1):
        title = item.get("title") or "(无标题)"
        content = item.get("content") or ""
        url = item.get("url") or ""
        lines.append(f"{index}. {title} — {content}")
        lines.append(f"   URL: {url}")
    lines.append("若需正文，请用 fetch 打开其中某个 URL。")
    return "\n".join(lines)


def format_fetched_content(url, raw_content) -> str:
    """把抓到的正文排版成给模型看的字符串，超长截断。"""
    body = raw_content or ""
    if len(body) > MAX_FETCH_CHARS:
        return (
            f"正文（{url}）：\n"
            + body[:MAX_FETCH_CHARS]
            + f"\n...[正文已截断，共 {len(body)} 字]"
        )
    return f"正文（{url}）：\n" + body


# =========================
# 网络层（需要 key 和网络）
# =========================


def search_items(query, top_k=3):
    """返回原始结果 dict 列表（编排器要用到 URL 和结构化内容）。

    search() 是它的「排版版」；deep_research 编排器直接用它拿 URL，再自己调
    format_search_results 排版喂给 LLM。
    """
    client = _client()  # 缺 key 在这里立刻抛，不进入重试

    def do_search():
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=top_k,
            include_answer=False,
        )
        results = response.get("results", [])
        print(f"[Tavily]   search({query!r}) → 返回 {len(results)} 条")
        return results

    return _with_retry(do_search)


def search(query, top_k=3):
    return format_search_results(query, search_items(query, top_k))


def fetch(url):
    client = _client()

    def do_fetch():
        response = client.extract(urls=[url])
        results = response.get("results", [])
        failed = response.get("failed_results", [])

        if not results:
            if failed:
                # 预期内的失败：反爬、需登录、链接失效等，给模型一句能看懂的话。
                print(f"[Tavily]   fetch({url!r}) → 失败（被反爬/需登录/已失效）")
                return f"Error: 无法抓取 {url!r}（可能被反爬、需登录或已失效）"
            print(f"[Tavily]   fetch({url!r}) → 无正文")
            return f"Error: 抓取 {url!r} 未返回任何正文"

        raw = results[0].get("raw_content", "")
        print(f"[Tavily]   fetch({url!r}) → 正文 {len(raw)} 字")
        return format_fetched_content(url, raw)

    return _with_retry(do_fetch)
