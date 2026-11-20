# 可验证奖励强化学习中的答案验证器可靠性研究

**Answer Verification in RLVR**

本项目研究数值答案验证器如何影响可验证奖励强化学习（RLVR）的训练信号和最终答题表现。实验使用 GSM8K，先在 Qwen2.5-0.5B-Instruct 上建立基线，后续以 Qwen2.5-1.5B-Instruct 开展错奖纠正实验。

## 研究问题与实验进展

当标准答案是 `42` 时，回答「考虑过 42，最终答案是 `<answer>7</answer>`」会通过全文数值匹配，严格的最终答案验证器给 0 分。GRPO 会在同题的多条回答之间计算相对优势，因此一笔错奖可能改变整组回答的训练信号。

| 阶段 | 对照设计 | 代码入口 |
| --- | --- | --- |
| E1 | 用 GSM8K 示范做 LoRA SFT，从同一模型起点建立后续对照 | [SFT 数据与掩码](src/verifierlab/sft.py) |
| E2 / E3 | 从同一个 E1 出发，分别使用严格和宽松验证器做 GRPO | [训练入口](src/verifierlab/runner.py) |
| E4 | 保留组内奖励分布，在格式得分相同的严格失败回答间随机重分配宽松奖励 | [奖励重分配](src/verifierlab/rewards.py) |
| E5 | 等撤奖次数比较「有对有错」组和「四条全错」组，并保留原宽松规则和最终答案优先作为对照 | [纠错策略](src/verifierlab/repair_policy.py) · [实验配置](configs/gsm8k-1.5b-reward-repair/repair.json) |
| 固定回答分析 | 重算不同纠错方式下的优势和局部 LoRA 梯度 | [梯度测量](src/verifierlab/repair_gradient.py) · [预算分配回放](src/verifierlab/repair_allocation.py) |

### 1.5B 基线与在线对照

1.5B 的 E1–E4 在筛查后冻结的 200 道 GSM8K 测试题上每题生成四个回答。E1、E2、E3、E4 的最终答案正确率分别为 **49.38%、50.75%、51.00%、52.63%**。E2–E4 各训练 100 步，使用同一个 GRPO 种子。早期 0.5B 的贪心评测和 300 步探索单独保存在[历史单卡结果](docs/results/single-gpu-baseline.md)。

E5 在筛选出的 35 道训练题上，对每个种子分别训练四种策略，每种 100 步。两种限额策略在每个种子中各撤回 11 笔错奖，且每 25 步的撤奖数相同。下表使用同一批 200 道冻结题、每题四个回答的最终答案正确率。

| 对照 | 种子 47 | 种子 48 | 平均差 |
| --- | ---: | ---: | ---: |
| 全错组纠错 → 有对有错组纠错 | 50.00% → 52.50% | 53.75% → 52.50% | +0.63 个百分点 |
| 原宽松规则 → 最终答案优先 | 52.25% → 47.75% | 53.00% → 49.50% | −4.00 个百分点 |

第一项的按题配对 95% 区间为 −2.06 到 +3.31 个百分点，两个种子的差值方向相反，目前无法确定哪类组应优先纠错。

在固定的 1,728 条回答上，最终答案优先规则在复核后将错奖数从 92 降到 2。梯度测量覆盖 55 个有错奖且最终答案可判定的组，其中 8 组的平均梯度偏差在单笔撤奖后增大。在 20 个有多笔错奖的组中，将相同的 17 笔撤奖集中到若干组做完整纠错，局部梯度与参考梯度的平均距离比逐笔分配低约 26%。整组纠错的在线训练效果仍待评估。 [研究设计](docs/design.md)

## 查看复现步骤

从仓库根目录使用 Python 3.11 或更新版本查看命令。下面只打印 1.5B 在线对照的步骤，路径参数可按实际实验目录替换。

```bash
PYTHONPATH=src python3 -m verifierlab.reproduce preview online \
  --work-dir outputs/repair --baseline-dir outputs/baseline \
  --preparation-dir outputs/baseline/preparation --model-dir /path/to/qwen-1.5b-snapshot
```

[实验手册](docs/experiment.md#复现当前实验)列出完整运行顺序、模型和数据准备、各阶段的 smoke 审阅及归档分析。正式训练环境为 Ubuntu x86_64、Python 3.11、兼容 CUDA 12.8 的驱动和一张支持 BF16 的 NVIDIA GPU。依赖以[锁定环境](requirements/README.md)为准。

## 仓库结构与贡献

| 路径 | 内容 |
| --- | --- |
| [configs/](configs/) | 固定模型与数据版本、通用训练设置、评测和测试集筛查 |
| [configs/gsm8k-1.5b-reward-repair/](configs/gsm8k-1.5b-reward-repair/) | 纠错实验专用的抽样、筛查、分组、撤奖预算和归档判定 |
| [src/verifierlab/](src/verifierlab/) | [复现计划](src/verifierlab/reproduce.py)、训练、评测、证据校验与[结果分析](src/verifierlab/study_analysis.py) |
| [tests/](tests/) | 验证器、数据处理和实验流程的回归测试 |
| [docs/](docs/) | [研究设计](docs/design.md)、[代码架构](docs/architecture.md)、[技术规格](docs/spec.md)、实验手册和历史结果 |

贡献者可按[开发与验证流程](CONTRIBUTING.md)审阅实验配置和代码。模型权重、原始回答及逐条运行日志保存在实验归档中。

## 许可证

本仓库的原创代码和文档采用 [MIT 许可证](LICENSE)。模型、数据集和第三方依赖适用各自的许可证。模型与数据集的来源和许可证见 [0.5B 输入配置](configs/sources.json) 和 [1.5B 输入配置](configs/sources-1.5b.json)。

## 联系作者

作者为 [HighsunYvY](https://github.com/HighsunYvY)。项目问题、复现反馈和建议可通过 [GitHub Issues](https://github.com/HighsunYvY/rlvr-answer-verification/issues) 提交。如有其他问题或合作意向，欢迎通过邮件联系 [fenggaoyang910@gmail.com](mailto:fenggaoyang910@gmail.com)。
