"""广度 × 深度 的研究编排器（递归树搜索，借鉴 deep-research 项目）。

核心思想（两个参数）：
- 广度 breadth：每一层把当前问题拆成多少个**不同角度**的检索 query，并行去搜。
- 深度 depth：每搜完一层，从结果里提炼 learnings（要点）+ follow-up questions
  （后续问题），再用后续问题当新 query 往下递归一层，同时 breadth 减半
  （越深越聚焦）、depth 减 1，直到 depth=0 触底。

流程（对应 deep-research.ts 的四个函数）：
  1. generate_serp_queries：把问题拆成 breadth 个 query，每个带 research_goal
  2. 并行检索每个 query → process_search_result 提炼 learnings + follow_up_questions
  3. 每个分支用 follow_up_questions 当新 query，breadth=ceil(breadth/2)、depth-1 递归
  4. 触底后整棵树的 learnings / URLs 去重汇总 → write_final_report 写报告

与单步 agent.py 的 run() 不同：这里不是「让模型在一个 loop 里自己决定搜几次」，
而是「由代码显式控制搜多宽、挖多深」，模型只负责拆 query / 提炼 / 写报告三步。
"""

import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import search_backend
from llm import client
from openai import APIConnectionError, APITimeoutError


# =========================
# 配置
# =========================

# 同层并行发起的检索数上限。Tavily 免费档容易限流，默认 2 比较稳，可在 .env 调。
RESEARCH_CONCURRENCY = int(os.getenv("RESEARCH_CONCURRENCY", "2"))
# 每次 search 取前几条结果（参考实现用 limit=5）
SEARCH_TOP_K = 5
# 每个 query 结果默认提炼几条要点
NUM_LEARNINGS = 3

# 网络重试：与 agent.py / search_backend.py 同样的策略 —— 瞬时错误重试几次。
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
RETRY_DELAY_SECONDS = float(os.getenv("RETRY_DELAY_SECONDS", "1.0"))


SYSTEM_PROMPT = (
    "你是一个资深研究助理，负责拆解检索方向、从资料中提炼要点、撰写研究报告。"
    "要点要信息密度高，包含实体（人名/机构/产品/作品等）和确切的数字、日期。"
    "只依据提供的检索结果，禁止编造。"
)


# =========================
# LLM 调用层
# =========================

def _create(messages, response_format=None):
    """调 DeepSeek，瞬时网络错误自动重试。"""
    last_exc = None
    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            kwargs = dict(
                model=os.getenv("DEEPSEEK_MODEL"),
                messages=messages,
                extra_body={"thinking": {"type": "disabled"}},
            )
            if response_format:
                kwargs["response_format"] = response_format
            return client.chat.completions.create(**kwargs)
        except (APITimeoutError, APIConnectionError) as exc:
            last_exc = exc
            if attempt < RETRY_MAX_ATTEMPTS - 1:
                print(f"[DeepSeek] 网络异常（{type(exc).__name__}），{RETRY_DELAY_SECONDS}s 后重试...")
                time.sleep(RETRY_DELAY_SECONDS)
    raise last_exc


def _parse_json(text):
    """把模型返回解析成 dict，兜底抠第一个 {...}（模型偶尔会包说明文字或代码块）。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _call_llm_json(prompt):
    """结构化输出：要求模型只回 JSON，并解析成 dict。"""
    resp = _create(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    return _parse_json(resp.choices[0].message.content)


def _call_llm_text(prompt):
    """自由文本输出（写报告用），直接返回正文。"""
    resp = _create(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


# =========================
# 广度 × 深度 四件套
# =========================

def generate_serp_queries(query, num_queries, learnings=None):
    """把研究主题拆成至多 num_queries 个不同角度的检索 query。"""
    parts = [
        "根据用户的研究主题，生成用于检索的 query 列表（最多 " + str(num_queries) + " 个，",
        "如果原问题已经很清晰可以少于 " + str(num_queries) + " 个）。",
        "每个 query 必须角度不同、彼此不重复。",
        "",
        "<主题>" + query + "</主题>",
    ]
    if learnings:
        parts.append("")
        parts.append("以下是此前研究得到的要点，据此生成更具体、更深层的问题：")
        for l in learnings:
            parts.append("- " + l)
    parts.append("")
    parts.append(
        "请严格按 JSON 输出，不要输出任何其他文字："
        '{"queries": [{"query": "检索关键词", "research_goal": "这个 query 想达成的研究目标；'
        '先说目标，再说拿到结果后如何进一步推进、还有哪些补充研究方向，尽量具体"}]}'
    )
    data = _call_llm_json("\n".join(parts))
    queries = data.get("queries") or []
    print(f"[Research] 拆出 {len(queries)} 个检索方向")
    return queries[:num_queries]


def process_search_result(query, result_text, num_learnings=NUM_LEARNINGS, num_follow_up_questions=3):
    """从一次检索结果里提炼 learnings 和 follow-up questions。"""
    prompt = (
        "下面是针对 query「" + query + "」的检索结果。请提炼出至多 " + str(num_learnings) + " 条要点（learnings），"
        "每条要简洁、信息密度高，包含实体（人/机构/产品/作品等）和确切的数字、日期。"
        "再给出至多 " + str(num_follow_up_questions) + " 个后续研究问题，用于继续深挖。\n"
        "<检索结果>" + result_text + "</检索结果>\n\n"
        '请严格按 JSON 输出，不要输出任何其他文字：{"learnings": ["要点1", ...], "follow_up_questions": ["问题1", ...]}'
    )
    data = _call_llm_json(prompt)
    learnings = data.get("learnings") or []
    follow_ups = data.get("follow_up_questions") or []
    print(f"[Research] {query!r} → {len(learnings)} 条要点、{len(follow_ups)} 个后续问题")
    return {"learnings": learnings, "follow_up_questions": follow_ups}


def deep_research(query, breadth, depth, learnings=None, visited_urls=None):
    """递归树搜索：一层搜 breadth 个方向，逐层收窄（breadth 减半）往下钻 depth 层。

    返回 {"learnings": [...去重], "visited_urls": [...去重]}。
    """
    learnings = list(learnings or [])
    visited_urls = list(visited_urls or [])

    serp_queries = generate_serp_queries(query, breadth, learnings)
    if not serp_queries:
        return {"learnings": learnings, "visited_urls": visited_urls}

    def _branch(serp_query):
        new_breadth = math.ceil(breadth / 2)   # 越深越聚焦：宽度减半
        new_depth = depth - 1

        try:
            items = search_backend.search_items(serp_query["query"], top_k=SEARCH_TOP_K)
            result_text = search_backend.format_search_results(serp_query["query"], items)
            new_urls = [it.get("url") for it in items if it.get("url")]
        except Exception as exc:
            # 单条检索失败不该丢掉整条分支已积累的成果（参考实现这里会丢，刻意改进）
            print(f"[Research] 检索失败 {serp_query['query']!r}: {exc}")
            return {"learnings": learnings, "visited_urls": visited_urls}

        processed = process_search_result(
            serp_query["query"], result_text,
            num_follow_up_questions=new_breadth,
        )
        branch_learnings = learnings + processed["learnings"]
        branch_urls = visited_urls + new_urls

        if new_depth > 0:
            next_query = (
                "此前研究目标：" + serp_query.get("research_goal", "") + "\n"
                "后续研究方向：\n"
                + "\n".join("- " + q for q in processed["follow_up_questions"])
            )
            print(f"[Research] 下钻一层 breadth={new_breadth} depth={new_depth}")
            return deep_research(
                query=next_query,
                breadth=new_breadth,
                depth=new_depth,
                learnings=branch_learnings,
                visited_urls=branch_urls,
            )
        return {"learnings": branch_learnings, "visited_urls": branch_urls}

    # 同一层的多个 query 并行搜
    with ThreadPoolExecutor(max_workers=RESEARCH_CONCURRENCY) as pool:
        results = list(pool.map(_branch, serp_queries))

    return {
        "learnings": _dedupe([l for r in results for l in r["learnings"]]),
        "visited_urls": _dedupe([u for r in results for u in r["visited_urls"]]),
    }


def write_final_report(prompt, learnings, visited_urls):
    """用整棵树汇总的 learnings 写一份 Markdown 报告，末尾附来源。"""
    learnings_str = "\n".join("<learning>" + l + "</learning>" for l in learnings)
    text = _call_llm_text(
        "根据用户的研究主题，用研究中得到的要点写一份详尽的最终报告（Markdown），"
        "尽可能详细、覆盖所有要点、结构清晰。\n\n"
        "<主题>" + prompt + "</主题>\n\n"
        "以下是研究得到的所有要点：\n<learnings>\n" + learnings_str + "\n</learnings>"
    )
    urls_section = "\n\n## Sources\n\n" + "\n".join("- " + u for u in visited_urls)
    return text + urls_section


def _dedupe(items):
    """去重且保序。"""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
