# 代码架构

仓库的 Python 包位于 [src/verifierlab](../src/verifierlab/)。命令通过 Python 模块执行。Transformers 负责模型加载与生成，TRL 负责训练循环，PEFT 负责 adapter 注入和合并。项目代码负责冻结输入、定义奖励、校验实验身份、保存逐样本证据及统计分析。当前 GPU 入口支持单进程和单张 NVIDIA GPU。

| 环节 | 实现 | 数据流与职责 |
|---|---|---|
| 准备输入 | [snapshots.py](../src/verifierlab/snapshots.py)、[data.py](../src/verifierlab/data.py) | 校验固定版本的模型与 Parquet，解析 GSM8K，写入四个分片及 manifest。记录保留样本 ID、提示词、gold 和 SFT 示范。 |
| 分发命令 | [runner.py](../src/verifierlab/runner.py)、[experiment.py](../src/verifierlab/experiment.py) | 统一入口分发训练、评测、审计、准入和分析。运行会话验证输入、环境与 checkpoint，建立运行签名，记录产物和最终状态。 |
| 复现计划 | [reproduce.py](../src/verifierlab/reproduce.py)、[configs/gsm8k-1.5b-reward-repair/](../configs/gsm8k-1.5b-reward-repair/) | 打印训练和分析命令，从已审计的全错条件生成有对有错条件的分窗口预算。 |
| 准备 SFT | [sft.py](../src/verifierlab/sft.py)、[training.py](../src/verifierlab/training.py) | 将提示词和示范转换成 token 与回答掩码，核查 collator 实际生成的 labels，再交给 TRL。 |
| 训练 GRPO | [training.py](../src/verifierlab/training.py)、[rewards.py](../src/verifierlab/rewards.py)、[repair_policy.py](../src/verifierlab/repair_policy.py) | 合并 E1 后注入新 adapter。奖励回调核对组内顺序和 gold，记录分项奖励与 E4 重分配或 E5 干预。 |
| 生成与评分 | [rollouts.py](../src/verifierlab/rollouts.py)、[generate.py](../src/verifierlab/generate.py)、[evaluate.py](../src/verifierlab/evaluate.py) | 从同一冻结输入读取提示词，生成并切出新 token，按样本 ID 关联 gold，输出逐条判定和汇总指标。E1 审计使用分组采样。 |
| 定义判定 | [verifier.py](../src/verifierlab/verifier.py) | 共用数值归一化，分别执行末答标签选择、全文数值候选选择和格式判定。训练回调与离线评分复用这一实现。 |
| 分析与旧产物审计 | [analysis.py](../src/verifierlab/analysis.py)、[audit.py](../src/verifierlab/audit.py) | 配对分析校验冻结测试 ID、gold、解码参数、E1 谱系和 E2/E3 控制变量。旧版 E0 审计从 token 重新解码、评分并校验产物一致性。 |
| 错奖研究分析 | [repair_study.py](../src/verifierlab/repair_study.py)、[study_analysis.py](../src/verifierlab/study_analysis.py)、[repair_gradient.py](../src/verifierlab/repair_gradient.py)、[repair_allocation.py](../src/verifierlab/repair_allocation.py) | 配对比较 E5 测试输出，汇总固定回答的错奖和梯度变化，在相同预算下分析整组与逐笔纠错。 |
| 环境与本地辅助 | [preflight.py](../src/verifierlab/preflight.py)、[cli.py](../src/verifierlab/cli.py)、[utils.py](../src/verifierlab/utils.py) | 环境模块检查固定依赖和可选 BF16 CUDA 运算。本地工具处理 JSONL、预览配置和校验批次结构。 |

## 训练与 checkpoint

SFT 先分别对提示词和完整对话应用 chat template，再验证提示词 token 是完整序列的精确前缀。提示词与 padding 的 labels 设为 `-100`，回答 token 与结束标记进入交叉熵目标。超长示范触发失败。实际 collator 掩码保存在 `masks.json`。

GRPO 每个提示词采样四个回答，按组缩放奖励。奖励回调读取冻结 gold 元数据，生成端只接收 prompt。禁用新 RL adapter 后，PEFT 参考路径回到合并后的 E1；固定输入上的前向对照验证这一语义。TRL 0.24.0 的首次 GRPO 验证读取梯度累积步数，项目在这次调用期间提供配置值，随后把累积状态交还训练循环。

训练保存 adapter、checkpoint 谱系和训练契约。重载检查对比参数与同一输入上的 logits，两次前向共用 CUDA BF16 autocast、eval 模式和 attention mask，并关闭 KV cache。项目保留误差统计和原始对照张量。

## 运行记录与入口

统一实验入口 `python -m verifierlab.runner` 产生 schema 2 运行记录。运行签名由阶段、profile、设置、输入、策略、依赖和 GPU 环境共同确定。`verifierlab.generate` 和 `verifierlab.audit` 分别提供历史 schema 1 的生成和审计入口。`verifierlab.cli` 提供本地配置预览、JSONL 数据处理、离线评分和资源诊断。

会话以独立目录保存 `run.json` 和产物清单。生成按条写入原始 token 与评分记录。训练记录梯度、参数变化、验证指标、重载结果和 GPU 资源。逐样本数据和运行证据的操作流程见[实验手册](experiment.md)。
