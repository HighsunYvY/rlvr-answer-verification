# 规格与项目约束

## 输入、配置和训练契约

- [sources.json](../configs/sources.json) 固定 Qwen2.5-0.5B-Instruct 模型与 tokenizer revision、GSM8K main revision 和文件清单。输入准备验证上游对象及本地字节身份。实验使用经过验证的本地快照和分片。
- `sample_id` 采用 `train:索引` 或 `test:索引`。GSM8K 原始答案使用单个 `####` 分隔示范推导和数值 gold。数据集保存各字段，模型生成只接收提示词，SFT 使用示范，GRPO 奖励和评测通过冻结记录取 gold。
- [data.yaml](../configs/data.yaml)、[sft.yaml](../configs/sft.yaml)、[grpo_strict.yaml](../configs/grpo_strict.yaml)、[grpo_weak.yaml](../configs/grpo_weak.yaml) 和 [evaluation.yaml](../configs/evaluation.yaml) 使用 JSON 语法。实际参数以锁定的配置和运行记录为准。默认 LoRA 为 `q_proj/v_proj`、秩 16、alpha 32、dropout 0。SFT 峰值学习率为 `2e-4`。GRPO 峰值学习率为 `5e-6`，使用组内缩放、`G=4`、KL 系数 0.04、temperature 0.8、top-p 0.95 和 256 token 的提示词上限。
- 当前训练契约为单进程 BF16 CUDA。微批大小为 1，梯度累积为 8；默认 GRPO 生成批含 8 个回答，对应两个各含 4 个回答的提示词组。

## 验证器 v1

strict 与 weak 都将合法数值通过 `Decimal` 转为 `Fraction`，按精确有理数比较。输入接受 ASCII 数字、正负号（含 Unicode 负号）、小数、规范三位千分位、科学计数法和简单分数。斜杠两侧允许空白。去掉首尾空白后长度上限为 128，十进制表示的指数绝对值上限为 100。完整解析会拒绝零分母、非有限值、单位、货币、百分号和表达式。

- **strict** 选择最后一个语法合法的 `<answer>…</answer>`。标签区分大小写，要求闭合、非空、无嵌套，内容限于无角括号的文本。所选内容必须完整通过数值解析。末个合法标签内容解析失败时得 0 分。后续未闭合标签保留先前合法标签的可选资格。
- **format** 要求全文恰好出现一对答案标签，标签内容是合法数值，结束标签后只允许空白。此前允许有正文。格式与数值正确性分别计分。
- **weak** 扫描新生成的完整 assistant 回答，包括答案标签内容。实现先处理数字后接空白、文本结尾或 `<` 前的句末逗号和句点，再按 [verifier.py](../src/verifierlab/verifier.py) 的 `_TOKEN` 提取完整数值候选。Unicode 单词字符、数值连接符及相邻数字参与边界约束。乘号、货币符号和百分号允许形成独立的数值候选，因此完整数值解析与全文提取具有不同的接受范围。

| 回答 | Gold | Strict | Weak | Format |
|---|---:|---:|---:|---:|
| `<answer>42</answer>` | 42 | 1 | 1 | 1 |
| `42 then <answer>7</answer>` | 42 | 0 | 1 | 1 |
| `<answer>42</answer><answer>oops</answer>` | 42 | 0 | 1 | 0 |
| `1/2` | 0.5 | 0 | 1 | 0 |
| `x42`、`142`、`42+2` | 42 | 0 | 0 | 0 |
| `42*2`、`$42`、`42%` | 42 | 0 | 1 | 0 |

奖励为 `0.2 × format + correctness`，两个分项各取 0 或 1。格式合规与正确性分别贡献奖励。批次错位触发失败，输入准备拒绝无效 gold。回归用例集中在 [test_verifierlab.py](../tests/test_verifierlab.py)。

## 证据与变更

评测记录样本 ID、生成文本、gold、解析答案、判定标志、奖励、错误类别、token 数和终止原因。生成 token 数包含 EOS，排除 EOS 后的 padding。EOS 与长度上限同时出现时以 EOS 为终止原因。缺少 EOS 且用完预算时标为截断。

令 `S` 表示 strict 通过，`W` 表示 weak 通过，`FP` 表示 weak 通过且 strict 失败。strict 正确率、格式率和 weak 阳性率以全部样本为分母。weak 分歧报告三种比例，即 `FP/全部样本`、`FP/strict 失败样本` 和 `FP/weak 通过样本`。分母为零的指标记为 `null`。

E3 准入在 E1 的完整开发集上采样。当前 [weak_audit.json](../configs/weak_audit.json) 要求至少 1 个 FP，至少 2 个 weak 总奖励有变化的组，且这类组占比至少为 0.1。审阅同时查看正确性奖励与总奖励的组内方差。达到阈值并完成证据审阅后，生成显式准入记录。变更验证器语义时更新规格版本并重新执行审计和配对实验。

本地开发与审阅使用 macOS，GPU 实验在 Ubuntu + CUDA 执行。模型与数据下载、环境变更、运行验证、提交和推送各自遵循项目授权范围。仓库保留代码、配置、文档和小型结果摘要；模型、数据、缓存、凭据、原始运行和个人指导材料保存在本地或持久存储。每次实验使用已提交版本和独立输出目录，保留失败及中间产物。协作要求见 [AGENTS.md](../AGENTS.md) 和 [CONTRIBUTING.md](../CONTRIBUTING.md)。
