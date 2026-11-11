import json
import os
import time

from llm import client
from openai import APIConnectionError, APITimeoutError
from tools import execute_tool, tool_schemas


# =========================
# 1. 给 Agent 立规矩
# =========================
#
# research 场景和 calculator 场景最大的区别：答案对不对，取决于它有没有真的
# 去查。所以 system prompt 的核心不是「怎么答」，而是「先查再答、必须标来源、
# 查不到就说查不到」—— 这三条正好对应模型最容易犯的三个毛病：凭记忆编、
# 不标出处、硬凑答案。

SYSTEM_PROMPT = (
    "你是一个研究助理，可以调用工具来保证结论可靠。\n"
    "工作方式：\n"
    "1. 需要事实资料时，先用 search 检索，从返回的摘要里挑出相关的 URL，再用 fetch 读正文；\n"
    "2. search 的 query 里若有人名、机构名、作品名等专有名词，必须原样保留，"
    "禁止改写、省略、音译或拆字；不确定时直接用用户原话里的词；\n"
    "3. 同一问题尽量从不同角度发 2~3 次 search（换关键词），并 fetch 至少两篇不同来源，"
    "结论一致才敢下，互相矛盾时如实说明分歧；\n"
    "4. 优先采用官方、一手来源；\n"
    "5. 需要算术时用 calculator；\n"
    "6. 结论只能来自工具返回的内容；禁止编造来源，也禁止凭记忆补细节；\n"
    "7. 检索不到相关内容就直接说找不到，不要硬凑。\n"
    "最终回答用这个结构：先给一句话结论，再分点列出论据，最后附上「来源」列表（每条用到的资料都给 URL）。\n"
    "简单问候、闲聊不需要调用工具，直接回答即可。"
)


# =========================
# 2. Agent Loop
# =========================

MAX_STEPS = 10

# 网络重试：代理（Clash）偶发断连会导致 DeepSeek 调用超时/连接失败，
# 与 Tavily 侧同样的处理 —— 瞬时错误重试几次，认证类错误不重试。
# 次数与间隔可在 .env 里配置。
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
RETRY_DELAY_SECONDS = float(os.getenv("RETRY_DELAY_SECONDS", "1.0"))


def _call_llm(messages):
    """调 DeepSeek，对瞬时网络错误（超时/连接失败）自动重试。"""
    last_exc = None
    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            return client.chat.completions.create(
                model=os.getenv("DEEPSEEK_MODEL"),
                messages=messages,
                tools=tool_schemas(),
                extra_body={"thinking": {"type": "disabled"}},
            )
        except (APITimeoutError, APIConnectionError) as exc:
            last_exc = exc
            if attempt < RETRY_MAX_ATTEMPTS - 1:
                print(f"[DeepSeek] 网络异常（{type(exc).__name__}），{RETRY_DELAY_SECONDS}s 后重试...")
                time.sleep(RETRY_DELAY_SECONDS)
    raise last_exc


def run(user_input, max_steps=MAX_STEPS):
    """LLM 判断 → Tool Calling → 工具执行 → 结果返回 → 最终回答"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    # 防空转：模型有时会反复发起完全相同的调用，直到把步数耗光。
    # 记下每次调用的指纹，重复的直接回一句提示，不再真的执行。
    seen = set()

    for step in range(1, max_steps + 1):
        print(f"[DeepSeek] 第 {step} 轮：请求模型判断...")
        resp = _call_llm(messages)

        msg = resp.choices[0].message
        # 必须把 assistant 消息本身放回上下文：后面每条 role="tool" 消息都要
        # 引用它携带的 tool_call_id，缺了它下一轮请求会被 API 拒绝。
        messages.append(msg)

        # 退出条件：模型不再要求调用工具，说明可以给出最终回答了。
        if not msg.tool_calls:
            print("[DeepSeek] 不再调用工具，生成最终回答")
            return msg.content

        # 模型可能在一轮里返回多个 tool_call，必须全部执行、逐一回传。
        for call in msg.tool_calls:
            print(f"[DeepSeek] 决定调用 {call.function.name}({call.function.arguments})")
            try:
                args = json.loads(call.function.arguments)
            except json.JSONDecodeError as exc:
                result = f"Error: 参数不是合法 JSON: {exc}"
            else:
                fingerprint = (call.function.name, json.dumps(args, sort_keys=True))
                if fingerprint in seen:
                    result = "Error: 重复调用，请改用不同参数或直接作答"
                else:
                    seen.add(fingerprint)
                    result = execute_tool(call.function.name, args)

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    return f"已达到最大步数 {max_steps}，未能得出最终答案。"
