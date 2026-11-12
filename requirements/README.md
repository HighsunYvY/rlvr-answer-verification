# Linux Python 3.11 / cu128 基线环境锁

`linux-py311-cu128.txt` 固定 Linux x86_64、Python 3.11 和 PyTorch 2.8.0+cu128 的依赖组合。原始解析环境为 WSL2、CPython 3.11.17 和 uv 0.13.0。直接版本来自 `pyproject.toml`，构建后端版本由 `build.in` 固定。分析绘图依赖单独管理。

锁文件沿用原始解析的字节和发行包哈希。锁 SHA256 为 `49124d00007b54fdb0d59d32d58f5c55f4fc36bb6bc79ee5b8c3890578393402`。新主机根据解释器、驱动和 GPU 预检结果选择兼容的环境。

环境安装、复用及验证命令见[实验手册](../docs/experiment.md#gpu-主机与环境)。
editable 构建使用锁中已安装的 setuptools。普通包取自 PyPI，torch 取自 PyTorch cu128 index。
索引选择使用显式的 `--torch-backend cu128`，锁文件注释记录实际解析索引。

重新解析属于独立的环境变更。下列命令输出候选锁，审阅通过后再替换现用锁：

```bash
uv pip compile pyproject.toml requirements/build.in --extra train --extra dev --python .venv/bin/python --torch-backend cu128 --generate-hashes --emit-index-url --emit-index-annotation -o requirements/linux-py311-cu128-candidate.txt
```

锁定及索引规则见 [uv 依赖锁定](https://docs.astral.sh/uv/pip/compile/)和 [PyTorch 索引选择](https://docs.astral.sh/uv/guides/integration/pytorch/)。
