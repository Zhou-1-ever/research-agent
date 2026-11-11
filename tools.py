"""工具注册表：把「告诉模型的 schema」和「真正干活的 Python 函数」绑在一起。

加一个新工具 = 写一个函数 + 贴一个 @register 装饰器。agent.py 一行都不用改。

设计上有两条刻意的约束：

1. **异常一律不外抛**。`execute_tool` 统一兜住所有异常并转成字符串回传给模型，
   让它看到失败原因并自行纠正，而不是直接炸掉整个 Agent Loop。这条不变式放在
   注册表层，意味着以后新增的任何工具都自动继承；工具函数自己只管干正事。
2. **工具函数保持纯函数**：入参是模型给的参数，返回一个字符串。
   不做 IO 之外的副作用，方便直接单测。
"""

from dataclasses import dataclass
from typing import Any, Callable

import search_backend


# =========================
# 1. 注册表本体
# =========================


@dataclass(frozen=True)
class Tool:
    """一个工具 = schema 元信息 + 实现函数。"""

    name: str
    description: str
    parameters: dict  # 手写 JSON Schema，就是 OpenAI tools 参数需要的格式
    fn: Callable[..., Any]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# dict 天然保序（Python 3.7+），所以工具的注册顺序就是发给模型的顺序。
REGISTRY: dict[str, Tool] = {}


def register(name, description, parameters):
    def decorator(fn):
        REGISTRY[name] = Tool(name, description, parameters, fn)
        return fn

    return decorator


def tool_schemas() -> list[dict]:
    """导出所有工具的 schema，直接喂给 create(tools=...)。"""
    return [tool.schema() for tool in REGISTRY.values()]


def execute_tool(name, args) -> str:
    """执行模型请求的工具，任何异常都转成 Error 字符串回传。

    注意用 `fn(**args)` 调用：这样参数名写错、必填项缺失产生的 TypeError
    也会被同一层兜住、回传给模型，模型自己就能改对，不用我们预判。
    """
    tool = REGISTRY.get(name)
    if tool is None:
        return f"Error: 未知工具 {name!r}"
    try:
        return str(tool.fn(**args))
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"


# =========================
# 2. 具体工具
# =========================


@register(
    name="calculator",
    description="计算两个数字的乘积",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "第一个数字"},
            "b": {"type": "number", "description": "第二个数字"},
        },
        "required": ["a", "b"],
    },
)
def calculator(a, b):
    return a * b


@register(
    name="search",
    description=(
        "按关键词检索资料，返回候选文章的标题、摘要和 URL。"
        "需要正文时再用 fetch 打开其中某个 URL。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索关键词"},
            "top_k": {
                "type": "integer",
                "description": "返回几条结果，默认 3",
            },
        },
        "required": ["query"],
    },
)
def search(query, top_k=3):
    return search_backend.search(query, top_k)


@register(
    name="fetch",
    description="打开 search 返回的某个 URL，读取该文章正文。",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "要读取正文的 URL，必须来自 search 的返回结果",
            },
        },
        "required": ["url"],
    },
)
def fetch(url):
    return search_backend.fetch(url)
