# agentstart

一个 Python 写的 LLM Agent 练手项目：用 DeepSeek 做大脑，用 Tavily 做真实联网检索，从「会调工具」一路做到「广度 × 深度的深度研究」。

## 三个阶段

1. **Calculator**：最小 Agent Loop，工具只有 `calculator`，验证「模型 → 工具调用 → 执行 → 回填 → 最终回答」的闭环。
2. **单步 Research Agent**：工具扩到 `search` + `fetch` + `calculator`，靠 system prompt 让模型「先查再答、标来源、查不到就说查不到」。
3. **广度 × 深度深度研究**：把 deep-research 的递归树搜索思想搬进来，由代码显式控制「搜多宽（breadth）、挖多深（depth）」。

## 广度 × 深度是什么

把「做研究」建模成一棵递归搜索树，靠两个参数控制：

- **广度 breadth**：每一层把当前问题拆成 `breadth` 个不同角度的检索 query，并行去搜。
- **深度 depth**：每搜完一层，从结果里提炼 learnings（要点）+ follow-up questions（后续问题），再用后续问题当新 query **往下递归一层**，同时 `breadth = ceil(breadth/2)`（越深越聚焦）、`depth - 1`，直到 `depth = 0` 触底。

最后把整棵树上所有 learnings / 来源 URL 去重汇总，写一份带来源的 Markdown 报告。

```
deep_research(query, breadth, depth)
├─ generate_serp_queries()      拆成 breadth 个不同角度的 query
├─ 每个 query 并行：search → process_search_result()
│     └─ 提炼 learnings + follow_up_questions
├─ 每支递归：breadth = ceil(breadth/2)，depth = depth - 1
└─ 触底后去重 → write_final_report() 写报告
```

## 项目结构

| 文件 | 作用 |
|---|---|
| `main.py` | CLI 入口：单步问答 + `research` 深度研究模式 |
| `agent.py` | 单步 Agent Loop（工具调用闭环，含空转检测、网络重试） |
| `deep_research.py` | 广度 × 深度递归树搜索编排器 |
| `tools.py` | 工具注册表：calculator / search / fetch，异常统一兜底 |
| `search_backend.py` | 可插拔检索后端（当前接 Tavily），全项目唯一知道「数据从哪来」的文件 |
| `llm.py` | DeepSeek 客户端（OpenAI 兼容） |
| `test_agent.py` | 工具层 + 端到端测试 |
| `test_deep_research.py` | 深度研究的离线测试（mock，验证递归形状） |

## 安装

```bash
pip install -r requirements.txt
```

依赖：`openai`、`python-dotenv`、`tavily-python`。

## 配置

复制 `.env` 模板并填上你的 key（`.env` 已在 `.gitignore` 中）：

```bash
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
TAVILY_API_KEY=tvly-...
```

可选配置：

| 变量 | 默认 | 说明 |
|---|---|---|
| `RETRY_MAX_ATTEMPTS` | 3 | 网络瞬时错误重试次数 |
| `RETRY_DELAY_SECONDS` | 1.0 | 重试间隔（秒） |
| `RESEARCH_CONCURRENCY` | 2 | 深度研究同层并行检索数（Tavily 免费档易限流） |

> 注意：网络需要能访问 DeepSeek 与 Tavily。若走代理（如 Clash），确保已开启。

## 使用

```bash
python main.py
```

进入后有两种模式：

- **直接提问**：走单步 Agent Loop，自动决定是否调用工具并给出带来源的结论。

  ```text
  你想调研什么？> asyncio 和多线程怎么选
  ```

- **深度研究**：输入 `research` 或 `深度研究`，然后填主题、广度、深度，最终报告保存到 `report.md`。

  ```text
  你想调研什么？> research
  研究主题？> asyncio 适合什么场景
  广度 breadth（推荐 2-4，默认 2）> 2
  深度 depth（推荐 1-3，默认 1）> 1
  ```

退出：`quit` / `exit` / `q` / `退出`，或 Ctrl+C。

> breadth / depth 越大越贵：depth=2、breadth=3 会发 9 次检索 + 11 次模型调用。默认 `breadth=2, depth=1` 已足够先用起来。

## 测试

```bash
python test_agent.py          # 工具层（确定性）+ 端到端（真实调用）
python test_deep_research.py  # 深度研究（离线 mock，不联网）
```

`test_agent.py` 的端到端用例需要联网和有效 key；`test_deep_research.py` 完全离线，只验证递归形状（breadth 减半、depth 递减、learnings 去重）。

## 设计要点

- **换检索后端只改一个文件**：`search_backend.py` 对外只暴露 `search` / `search_items` / `fetch`，agent 与工具层自始至终只看到「字符串进、字符串出」。
- **异常一律不外抛**：`execute_tool` 统一兜住并转成字符串回传模型，让它自己纠正，而不是炸掉整个 Loop。
- **排版与联网分层**：`format_*` 是纯函数，离线可单测；联网逻辑薄薄一层。
