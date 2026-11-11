# 大模型后训练（Post-Training）研究报告

## 一、执行摘要

后训练已成为大模型能力成型的关键阶段。本报告基于检索到的资料，系统梳理后训练的方法定位框架、主流技术路线（SFT / RLHF / PPO / DPO / GRPO）、主要厂商的差异化实践（Qwen2.5、TULU 3、DeepSeek-V3、Llama 3）、开源训练框架格局与选型依据，以及典型实践案例与云化部署路径。

核心结论有三点：

1. **SFT 是所有大模型的必经环节**，RLHF/PPO 工程复杂、计算昂贵，DPO 是平衡效果与工程复杂度的主流选择，GRPO 结合可验证奖励已成为 2025 年后推理模型训练的主流范式。
2. **各厂商路线分化明显**：Qwen2.5 走 SFT + 两阶段 RL（PPO→DPO→GRPO），TULU 3 走 SFT→DPO→RLVR，DeepSeek-V3 走 DPO→PPO→GRPO 并引入 self-rewarding，Llama 3 则进行 6 轮迭代式后训练。
3. **框架选型上**，LLaMA-Factory（71.3k stars）与 Unsloth（64.3k stars）是当前生态中最受关注的两个项目，前者覆盖方法最全、显存门槛最低（7B 模型 4-bit QLoRA 最低 6GB），后者以提速与省显存见长。


## 二、后训练方法的定位框架

四种主流方法在“效果—工程复杂度”光谱上各有定位：

| 方法 | 核心机制 | 定位 |
|------|----------|------|
| **SFT** | 让模型学会做某件事 | 行为对齐的基础，几乎所有大模型的必经环节 |
| **RLHF / PPO** | 通过奖励信号让模型越做越好 | 工程复杂且计算昂贵，属顶级实验室专属 |
| **DPO** | 用偏好对直接优化策略、绕过 RL 循环 | 平衡效果与工程复杂度的主流选择 |
| **GRPO** | 结合可验证奖励 | 2025 年后推理模型训练的主流范式 |

这一框架为后续厂商路线与框架选型的分析提供了基本坐标。


## 三、各厂商后训练路线对比

检索资料显示，四家代表厂商的训练管线存在显著差异：

- **Qwen2.5**：SFT + 两阶段 RL，具体为 **PPO → DPO → GRPO**。
- **TULU 3**：**SFT → DPO → RLVR**。
- **DeepSeek-V3**：**DPO → PPO → GRPO**，并在 V3 中引入 **self-rewarding** 策略。
- **Llama 3**：**6 轮迭代式后训练**，每轮包含 Reward Modeling、Rejection Sampling、SFT、DPO 四个环节。

值得注意的是，Qwen 官方博客将后训练概括为 **SFT + RLHF（PPO）**，并已开源 **1.8B / 7B / 14B / 72B** 四款模型——这与 Qwen2.5 的 PPO→DPO→GRPO 管线形成对照，反映官方表述与具体版本实践之间的层次差异。


## 四、技术演进：从 RLHF 到 GRPO 及其回摆

### 4.1 RLHF 的三阶段结构与代价

RLHF 由三个阶段构成：

1. 先 **SFT** 得到初始策略；
2. 再基于人类偏好排序对 **(x, y_w, y_l)** 训练奖励模型；
3. 最后用 **PPO** 最大化奖励。

其代价被明确记录为：**四个模型需同时占用显存**、训练不稳定、超参数难以调节。这正是 PPO 被视为“顶级实验室专属”的原因。

### 4.2 DPO 的简化

DPO 经数学推导可**跳过显式奖励模型与 PPO 循环**，因此成为效果与工程复杂度之间的折中主流。

### 4.3 GRPO 的兴起与回摆

- **2024 年**，GRPO 因**省掉 critic** 而被广泛采用。
- **2025 年**，长程 Agentic 任务兴起后又向 **PPO/critic 回摆**，原因在于：GRPO 整条输出共用一个优势，**token 级信用分配最粗**。

这一“简化—回摆”的轨迹说明：方法选择高度依赖任务形态，而非单纯追求管线精简。


## 五、开源训练框架格局与选型

### 5.1 框架对比（2026 年七牛云对比及 AWS 博客）

按 Star 数从少到多排列：**ms-swift、Firefly、DeepSpeedExamples、unsloth、LLaMA-Factory、FastChat**。

关键数据：

| 框架 | Star 数 | 其他指标 |
|------|---------|----------|
| **LLaMA-Factory** | 71.3k | 7B 模型 4-bit QLoRA 最低 **6GB 显存**；覆盖预训练/SFT/RM/PPO/DPO/ORPO；提供 CLI / WebUI / Python 三种用法 |
| **Unsloth** | 64.3k | 5.7k forks；**500+ 模型**；**2 倍提速**、**70% 省显存**；MoE 加速 **12 倍** |
| **Axolotl** | 11.9k | — |

### 5.2 Firefly 项目细节

**Firefly（yangjianxin1/Firefly）** 是一个开源大模型训练项目：

- 支持**预训练、SFT、DPO** 和**全参 / LoRA / QLoRA**；
- 兼容 **Qwen2.5 / Qwen2 / Llama3 / Gemma / MiniCPM / Mixtral-8x7B** 等数十款模型；
- 集成 **Unsloth** 后的实测效果：
  - Qwen2：提速 **47.32%**、省 **39.13%** 显存（最低 **8.43GB**）；
  - Llama3-8B：提速 **44.35%**、省 **42.58%** 显存（最低 **7.75GB**）；
- 开源 **firefly-train-1.1M** 等指令数据集。


## 六、实践案例与云化部署

### 6.1 小模型全参微调的过拟合警示

知乎 **Qwen3-1.7B** 微调实战：用 Transformers 加载 bfloat16 权重，**SwanLab** 监控（适配 **30+ 框架**，含 LLaMA Factory、veRL）。使用 **2000 条数据全参微调 1 个 epoch 后即过拟合**——val loss 在第 2 轮上升。该案例提示：小模型全参微调的数据量与轮次控制需格外谨慎。

### 6.2 云化无代码平台

**AWS 中国团队**基于 **LLaMA-Factory on SageMaker** 推出无代码 **ModelHub** 平台：

- 支持**全球区与宁夏区**部署；
- 兼容 **Llama / Qwen / Yi / Baichuan / ChatGLM / Phi** 等模型。

这代表后训练能力正从脚本级工具向托管化、无代码化平台演进。


## 七、结论与选型建议

1. **方法层面**：以 SFT 为必经基础；资源有限时优先 DPO；追求推理能力且具备工程能力时采用 GRPO + 可验证奖励；长程 Agentic 任务应重新评估 PPO/critic 的价值。
2. **厂商路线层面**：Qwen2.5、TULU 3、DeepSeek-V3、Llama 3 的管线差异表明不存在唯一最优解，迭代式（Llama 3 的 6 轮）与阶段式（Qwen2.5 的两阶段 RL）各有取舍。
3. **框架层面**：追求方法覆盖广度与低显存门槛选 **LLaMA-Factory**（71.3k stars，6GB 起）；追求训练速度与显存效率选 **Unsloth**（64.3k stars，2 倍提速、70% 省显存、MoE 加速 12 倍）；需要多模型兼容与 Unsloth 集成收益可选 **Firefly**。
4. **实践层面**：小模型全参微调需警惕过拟合（Qwen3-1.7B 案例中 2000 条数据 1 个 epoch 即出现 val loss 反弹）；工程落地可借助 SwanLab 监控与 ModelHub 类无代码平台降低门槛。

---

*说明：本报告所有实体、数字与日期均严格取自提供的检索要点，未作外部补充或推断。*

## Sources

- https://deepseek.csdn.net/6a32085d10ee7a33f27e4d29.html
- https://www.bilibili.com/video/BV1HYBWBaEE3
- https://zhuanlan.zhihu.com/p/19647641182
- https://qiankunli.github.io/2023/12/18/llm_post_training.html
- https://www.youtube.com/watch?v=Z8eLQbuRKbI
- https://github.com/yangjianxin1/Firefly
- https://zhuanlan.zhihu.com/p/1903848838214705484
- https://aws.amazon.com/cn/blogs/china/practical-series-on-fine-tuning-large-language-models-part-two
- https://news.qiniu.com/archives/1778998742717
- https://qwenlm.github.io/zh/blog/qwen