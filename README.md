# 可验证奖励强化学习中的答案验证器可靠性研究

**Answer Verification in RLVR**

本项目研究数值答案验证器如何影响可验证奖励强化学习（RLVR）的训练信号和最终答题表现。项目以 Qwen2.5-0.5B-Instruct 和 GSM8K 为基线，提供从输入准备、LoRA 微调、GRPO 训练到评测和审计的复现流程。

## 实验设计与结果

严格验证器判定最后一个合法答案标签中的数值。宽松验证器从整个回答中寻找与标准答案等价的数值。例如，当标准答案是 `42`，回答 `考虑过 42，最终答案是 <answer>7</answer>` 会获得宽松正确性奖励，严格正确性奖励为零。

E2 和 E3 分别从同一个 E1 checkpoint 出发，使用相同的训练数据、步数、采样参数和格式奖励，仅改变正确性验证器。下表是单张 RTX 3090 上的历史基线结果。所有条件在同一份 200 题测试集上使用相同的严格判分，E2 和 E3 各训练 100 个优化步。

| 条件 | 方法                                                  | 严格判分答对题数 |
| ---- | ----------------------------------------------------- | ---------------: |
| E0   | 直接评测基础模型                                      |          2 / 200 |
| E1   | 使用 GSM8K 示范做 LoRA 监督微调，只对回答部分计算损失 |         61 / 200 |
| E2   | 从 E1 出发，使用严格验证器做 GRPO                     |         59 / 200 |
| E3   | 从同一 E1 出发，使用宽松验证器做 GRPO                 |         59 / 200 |

相对 E1，两种 GRPO 的严格正确率点估计均低 1 个百分点，配对 bootstrap 区间均跨越零。开发集审计观察到了两种验证器的判分分歧和组内奖励差异，策略是否学会系统性利用宽松规则仍待验证。详见[单卡实验结果](docs/results/single-gpu-baseline.md)，其中也记录了单独进行的 300 步开发集探索。

## 快速开始

在仓库根目录使用 Python 3.11 或更新版本。下面两条命令可以在 macOS 上离线查看输入清单和训练配置。

```bash
PYTHONPATH=src python3 -m verifierlab.snapshots --plan
PYTHONPATH=src python3 -m verifierlab.cli preview-config --config configs/grpo_strict.yaml --profile smoke
```

第一条命令列出固定版本的输入文件和预计大小。第二条命令展开指定规模下的 GRPO 参数。

正式训练使用 Ubuntu x86_64、Python 3.11、兼容 CUDA 12.8 的驱动和一张支持 BF16 的 NVIDIA GPU，执行入口要求 `WORLD_SIZE=1`。按[实验手册](docs/experiment.md)准备锁定环境和模型数据，依次执行环境预检、各阶段 smoke、审阅和正式运行。[依赖锁说明](requirements/README.md)记录了目标环境的版本约束。

## 仓库结构与贡献

| 路径                                 | 内容                                                                                                   |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| [configs/](configs/)                 | 模型与数据版本、数据划分、训练、评测和审计配置                                                         |
| [src/verifierlab/](src/verifierlab/) | 实验入口、训练、验证器、评测和证据校验实现                                                             |
| [tests/](tests/)                     | 验证器、数据处理和实验流程的回归测试                                                                   |
| [docs/](docs/)                       | [研究设计](docs/design.md)、[代码架构](docs/architecture.md)、[技术规格](docs/spec.md)、实验手册和结果 |

## 许可证

本仓库的原创代码和文档采用 [MIT 许可证](LICENSE)。模型、数据集和第三方依赖适用各自的许可证。模型与数据集的来源和许可证见 [输入配置](configs/sources.json)。

## 联系作者

作者为 [HighsunYvY](https://github.com/HighsunYvY)。项目问题、复现反馈和建议可通过 [GitHub Issues](https://github.com/HighsunYvY/rlvr-answer-verification/issues) 提交。如有其他问题或合作意向，欢迎通过邮件联系 [fenggaoyang910@gmail.com](mailto:fenggaoyang910@gmail.com)。
