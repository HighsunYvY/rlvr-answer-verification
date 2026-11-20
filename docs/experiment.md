# 实验手册

从仓库根目录执行本手册的命令。macOS 用于静态检查和获准的离线测试。Ubuntu + CUDA 执行端运行已提交版本。实验设计和控制变量见[研究设计](design.md)，验证器及记录字段见[规格](spec.md)。

## 本地检查

先用 `git status --short` 查看现有修改，审阅相关 diff、配置、文档链接和忽略规则。审查 Python 语法时可以只解析源码，检查 JSON 语法配置时可以只读取文本：

```bash
python3 - <<'PY'
import ast
import json
from pathlib import Path

for directory in ("src", "tests"):
    for path in Path(directory).rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
for path in Path("configs").rglob("*"):
    if path.is_file() and path.suffix in {".yaml", ".json"}:
        json.loads(path.read_text(encoding="utf-8"))
PY
git diff --check
git diff --cached --check
```

取得运行验证授权后，在 Mac 使用 Python 3.11 或更新版本运行合成回归。测试覆盖数据隔离、验证器、奖励、掩码和运行证据契约，使用标准库和合成输入。

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

提交和推送分别按 [CONTRIBUTING.md](../CONTRIBUTING.md) 审阅和授权。向 GPU 端交接完整提交 SHA、运行预算、命令和产物路径。

## GPU 主机与环境

使用具有持久存储的 Ubuntu x86_64 主机和一张支持 BF16 的 NVIDIA GPU。已有 checkout 先查看 `git status --short`。发现现场修改时先保留并处理，再同步。新主机使用 Git clone；已有干净主机使用 fast-forward。运行前对照交接 SHA：

```bash
git status --short
test -z "$(git status --porcelain)" &&
  git pull --ff-only origin main &&
  test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
```

将 `EXPECTED_SHA` 设为已审阅的完整提交 SHA，SHA 比对成功后继续。复用符合 [Linux 依赖锁](../requirements/README.md) 的项目独立环境。新建环境属于单独授权范围；首次准备时使用兼容的 Python 3.11 和 cu128 驱动：

```bash
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python --require-hashes --torch-backend cu128 requirements/linux-py311-cu128.txt
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .
```

将以下运行块放在同一个 Bash 会话中执行。每次使用独立根目录。`timeout` 由 Ubuntu 的 GNU coreutils 提供。按实际预算设置单条命令的时限。

```bash
set -euo pipefail
PY=.venv/bin/python
export PYTHONPATH=src
RUN_ROOT="outputs/$(date +%Y%m%d-%H%M%S)-full"
mkdir -p outputs
mkdir "$RUN_ROOT"
nvidia-smi | tee "$RUN_ROOT/nvidia-smi.txt"
"$PY" --version
uv pip check --python "$PY"
"$PY" -m pytest -q
.venv/bin/ruff check .
"$PY" -m verifierlab.preflight --check-trainer-api --probe-gpu --output "$RUN_ROOT/environment.json"
```

预检的通过范围是依赖、API 契约及 BF16 CUDA 运算和反向传播。真实生成与 Trainer 由后面的阶段检查验证。环境预检使用 `environment.json`，训练和生成使用各阶段目录的 `run.json`。历史 RTX 3090 证据见[单卡结果](results/single-gpu-baseline.md)，新主机的 GPU 结论以新产物为准。磁盘预留可从约 3 GB 输入空间、至少 5 GB 的运行证据空间及独立环境空间起算。根据实测峰值显存、吞吐、checkpoint 数量和剩余磁盘确定后续预算。

## 输入准备

| Profile | SFT / GRPO / 开发集 / 测试集 | 默认训练步数 | GRPO 回答上限 |
|---|---|---:|---:|
| `smoke` | 8 / 8 / 4 / 4 | 2 | 128 token |
| `pilot` | 64 / 64 / 32 / 32 | 20 | 256 token |
| `full` | 512 / 512 / 128 / 200 | 100 | 512 token |

`--profile` 选择冻结数据和预算。每个目标 profile 的 `--mode smoke` 使用同一份准备产物和序列上限，取八条训练样本和四条开发集或测试集样本进行检查，训练执行两步。`--mode run` 使用完整分片和声明预算。pilot 和 full 的正式运行需要各阶段匹配的真实 smoke 和人工审阅。

设置 `HF_HOME` 指向目标机的持久缓存。用 `"$PY" -m verifierlab.snapshots --sources configs/sources-1.5b.json --plan` 查阅固定文件清单与大小。以下示例从已有缓存准备 1.5B 的 full 输入。缓存缺失时，在获准的下载范围内另选输出目录，执行同一准备命令并加上 `--download`，下载时启用 Hub 网络访问，下载完成后恢复离线执行。准备 0.5B 输入时使用 `configs/sources.json`。

```bash
PROFILE=full
PREP="$RUN_ROOT/preparation"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" -m verifierlab.snapshots --sources configs/sources-1.5b.json \
  --profile "$PROFILE" --output-dir "$PREP"
MODEL="$("$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="passed"; print(r["snapshots"]["model"]["path"])' "$PREP/preparation.json")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

准备目录包含 `preparation.json` 和 `data/manifest.json`、`data/{sft,grpo,validation,test}.jsonl`。报告记录固定 revision、文件身份、行数和分片身份。每次准备的明细随运行产物保存在本地。

## Smoke、审阅和正式运行

下列函数把控制台日志保存在运行目录旁，并为每个阶段建立独立产物目录。pilot 和 full 的 `stage_pair` 在 smoke 后暂停，审阅者填写实际观察再继续。按阶段预算设置 `SMOKE_LIMIT` 和 `RUN_LIMIT`。示例默认值是超时上限，实际运行时间以日志为准。

```bash
experiment() { "$PY" -m verifierlab.runner "$@"; }

launch() {
  local limit=$1 name=$2 subcommand=$3
  shift 3
  timeout --signal=INT --kill-after=20s "$limit" \
    "$PY" -m verifierlab.runner "$subcommand" --profile "$PROFILE" \
    --preparation-dir "$PREP" --model-dir "$MODEL" \
    --output-dir "$RUN_ROOT/$name" "$@" 2>&1 | tee "$RUN_ROOT/$name.console.log"
}

review() {
  local action=$1 name=$2 evidence=$3 note
  printf 'Review %s/run.json and its artifacts. Enter the observations: ' "$evidence"
  read -r note
  test -n "$note" || return 1
  experiment "$action" --run-dir "$evidence" \
    --review-note "$note" --output "$RUN_ROOT/$name.json"
}

stage_pair() {
  local name=$1 subcommand=$2
  shift 2
  local admission=()
  launch "${SMOKE_LIMIT:-30m}" "$name-smoke" "$subcommand" --mode smoke "$@" || return
  if [ "$PROFILE" != smoke ]; then
    review admit "$name-admission" "$RUN_ROOT/$name-smoke" || return
    admission=(--admission "$RUN_ROOT/$name-admission.json")
  fi
  launch "${RUN_LIMIT:-6h}" "$name" "$subcommand" --mode run "${admission[@]}" "$@"
}
```

审阅者核对 `run.json` 的运行身份、产物清单和 GPU 记录。准入匹配阶段、profile、参数、模型与分片、策略、源码及依赖身份、驱动和 GPU。更换策略或评测分片时，为该对象执行对应的生成 smoke。

| 检查 | 审阅证据 |
|---|---|
| 生成 | 核对冻结 ID、原始生成 token、逐条评分、模板、EOS 和截断，以及吞吐和峰值显存。 |
| SFT | 核对真实 collator 的 prompt、padding、回答和 EOS 掩码，两个有限非零梯度、实际参数变化及 adapter 重载。 |
| GRPO | 核对每组四个回答及 gold 对齐、组内奖励变化、参考策略和 KL 记录。零梯度步保留在日志中。运行整体要求有限损失、有效训练组、至少一步有限非零梯度、参数变化和 adapter 重载。 |
| 重载 | 核对 adapter 参数一致性与固定输入的 logits 对照。相同 CUDA BF16 前向条件下使用 `atol=rtol=1e-4`，并记录最大绝对差和不匹配数。 |

环境探针、API 和合成测试各自覆盖前置条件。pilot/full 的准入证据来自目标阶段的真实模型运行。准确率用于结果分析，生成 smoke 以完整且一致的运行证据为通过条件。

## 复现当前实验

从仓库根目录预览某一阶段的命令。将 `--work-dir` 指向用于本轮产物的已有父目录，`--preparation-dir` 和 `--model-dir` 分别指向完整准备目录和本地模型快照。下面以同一个运行目录承载基线和后续实验为例。

```bash
"$PY" -m verifierlab.reproduce preview online --python "$PY" \
  --work-dir "$RUN_ROOT" --preparation-dir "$PREP" --model-dir "$MODEL"
```

替换 `online` 可以查看其他阶段。基线产物位于单独目录时，通过 `--baseline-dir` 指定基线产物的父目录。默认从该目录读取 `e1-train/` 和 `weak-admission.json`。历史 E1 目录名称不同时可用 `--sft-run` 指向实际路径。`gradient` 和 `offline` 默认读取 `--work-dir` 下的 `random-panel/`，也可通过 `--probe-dir` 使用归档中实际的采样目录。

| 阶段 | 输入与产物 |
| --- | --- |
| `baselines` | 准备审计过的 1.5B 分片，训练 E1–E4，生成冻结测试输出并审计 E4 奖励重分配。已有对应分片时直接使用。 |
| `probes` | 从同一个 E1 按两批冻结题目生成固定回答，按项目的筛选条件输出 `cohort.json`，与 [E5 配置](../configs/gsm8k-1.5b-reward-repair/repair.json) 中的 35 个训练 ID 对照。 |
| `online` | 使用冻结的 35 个 ID。每个种子先训练并审计全错策略，再根据实际撤奖数生成有对有错策略的计划。完成四个条件后生成冻结测试输出，统计两种配对差。 |
| `gradient` | 使用 `probes` 产出的随机采样回答和原 E1，计算固定回答的局部 LoRA 梯度。 |
| `offline` | 读取已完成的采样、梯度和冻结测试输出，统计错奖、梯度偏差以及整组和逐笔纠错的差异。 |
| `legacy` | 查看历史 0.5B 多种子流程。按打印的历史版本和对应准备产物运行，再回到当前版本汇总。重算原报告时，用已有的种子 42 训练和评测审计路径替换命令中的对应路径。 |

`preview` 只打印命令。执行时逐阶段核对真实 smoke，再填写生成的 `--review-note ''`；基于宽松规则的训练也需要审阅完整开发集的 E1 审计。E5 的对照表以审计中的实际 `withdrawn_awards` 验证预算匹配。任一窗口未达到配额时如实记录。

新生成的回答由自动判定汇总。重算原报告中复核后的错奖数时，将 `ARCHIVED_PROBE` 设为存有原始 `scored.jsonl` 和 `run.json` 的目录，并使用[归档人工判定](../configs/gsm8k-1.5b-reward-repair/archived-final-review.json)。在整组纠错的梯度回放中，梯度记录也必须来自同一份固定回答。

```bash
"$PY" -m verifierlab.study_analysis panel-rewards \
  --probe-dir "$ARCHIVED_PROBE" --preparation-dir "$PREP" \
  --review-file configs/gsm8k-1.5b-reward-repair/archived-final-review.json \
  --output "$RUN_ROOT/archived-panel-reward-counts.json"
```

## 历史 E0–E3 顺序

这一流程展示早期 0.5B 贪心评测的顺序。逐值复现历史表格时使用当时的代码和准备产物。先完成 E0 评测和 E1，再从同一 E1 开始 E2 与 E3。E3 先完成整个开发集的 E1 审计，并由审阅者核对 weak 分歧样本和组内奖励后生成单独准入。

```bash
stage_pair e0-eval evaluate --split test
stage_pair e1 train --stage E1
E1="$RUN_ROOT/e1"
stage_pair e1-eval evaluate --split test --checkpoint-dir "$E1"
stage_pair e2 train --stage E2 --sft-run "$E1"
stage_pair weak-audit weak-audit --checkpoint-dir "$E1"
review admit-weak weak-admission "$RUN_ROOT/weak-audit"
stage_pair e3 train --stage E3 --sft-run "$E1" \
  --weak-admission "$RUN_ROOT/weak-admission.json"
stage_pair e2-eval evaluate --split test --checkpoint-dir "$RUN_ROOT/e2" --sft-run "$E1"
stage_pair e3-eval evaluate --split test --checkpoint-dir "$RUN_ROOT/e3" --sft-run "$E1"
```

`admit-weak` 要求完整开发集、固定阈值和审阅记录。若审计低于阈值，保存审计与准入失败证据，E3 分支到此结束，其余已获准的分支可继续。训练初始和每十个优化步记录开发集诊断，最终策略以事先定义的预算确定。

复现 300 步开发集探索时，保留 full-profile 的 E1、准备产物和 weak 准入，另取独立输出名。以下命令在每次 smoke 和正式运行中传递相同的声明步数。100 步与 300 步的开发集评测分别使用各自 checkpoint 对应的生成准入。两步 smoke 仍只执行两个优化步。两次长训均从同一 E1 初始化，设置实际允许的长训时限：

```bash
stage_pair e2-300 train --stage E2 --sft-run "$E1" --max-steps 300
stage_pair e3-300 train --stage E3 --sft-run "$E1" --max-steps 300 \
  --weak-admission "$RUN_ROOT/weak-admission.json"
stage_pair e1-dev evaluate --split validation --checkpoint-dir "$E1"
for run in e2 e3 e2-300 e3-300; do
  stage_pair "$run-dev" evaluate --split validation \
    --checkpoint-dir "$RUN_ROOT/$run" --sft-run "$E1"
done
```

## 分析、恢复和备份

统一分析入口在 CPU 上校验冻结的测试集、配对关系和训练契约，输出 `summary.json`、`report.md`、`errors.jsonl` 和 `analysis.json`。开发集探索以冻结的开发集 ID 和同一解码配置单独配对统计。安装了 analysis extra 的环境可加 `--plot`，从实际指标和日志绘图。

```bash
experiment analyze --preparation-dir "$PREP" --output-dir "$RUN_ROOT/analysis" \
  --e0 "$RUN_ROOT/e0-eval" --e1 "$RUN_ROOT/e1-eval" \
  --e2 "$RUN_ROOT/e2-eval" --e3 "$RUN_ROOT/e3-eval" \
  --e1-training-run "$E1" --e2-training-run "$RUN_ROOT/e2" \
  --e3-training-run "$RUN_ROOT/e3" --weak-audit-run "$RUN_ROOT/weak-audit"
```

缺失某个 GRPO 条件时，省略其评测和训练参数，汇总保留该条件的缺失状态。早期 schema 1 的 E0 记录可用 CPU 重新解码和评分。将 `E0_LEGACY_RUN` 和 `E0_EXECUTION_SHA` 分别设为原始运行目录和原始执行提交。将 `PREP` 与 `MODEL` 设为该旧运行对应的 smoke 准备目录与本地模型快照：

```bash
"$PY" -m verifierlab.audit --run-dir "$E0_LEGACY_RUN" \
  --preparation-dir "$PREP" --model-dir "$MODEL" \
  --expected-code "$E0_EXECUTION_SHA" --output "$RUN_ROOT/e0-legacy-audit.json"
```

恢复训练在独立输出目录使用 `train --mode run --resume-from-checkpoint <原运行>/trainer/checkpoint-N`，沿用原阶段、签名、预算和准入。TRL 恢复训练状态；GRPO 从生成边界重新采样，结果可能与连续运行不同。中断恢复仍需专门运行验证。

释放实例前备份准备产物、原始生成、训练日志、环境记录、准入、父 E1 与 RL adapter、checkpoint 元数据及必要的训练状态。将 smoke、测试和正式运行日志写入 `outputs/` 或本地持久存储。保留失败和超时的控制台日志与已写入产物。迁移产物后，用完整原证据重新生成引用当前路径的准入记录。
