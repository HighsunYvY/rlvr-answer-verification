# Repository conventions

- Do not add or run GitHub Actions workflows. Execute local runtime checks only with user authorization; otherwise limit verification to static inspection.

- Treat VerifierLab as a reproducible post-training research system. Public artifacts describe implemented interfaces, protocols, evidence and limitations; keep personal mentoring, chat handoffs and career narratives out of them.
- Develop and review code on macOS; run models only on Ubuntu + CUDA rented or laboratory GPU hosts, with no local GPU/WSL validation path. After local checks and push, provide a revision-pinned server handoff for user execution and mark GPU evidence pending. GPU hosts are execution targets: check their working tree, fast-forward to an explicit committed revision, reuse the isolated environment, then run. Do not develop persistent remote forks, hardcode machine paths, or rebuild environments on every run.
- Keep E0–E3 controls intact and preserve per-sample evidence, failures and negative results. Use Transformers / TRL / PEFT; do not copy upstream repositories or replace Trainer loops.
- Model/data preparation, dependency changes and training are separate scoped operations. Honor existing authorization; do not implicitly download models, upgrade environments, start long training or provision paid/multi-GPU resources.
- Require a matching real model/Trainer GPU smoke before pilot/full. Environment probes, configuration checks and synthetic tests cannot substitute. Audit E1 development-set rollouts and within-group signal before weak-verifier GRPO.
- Preserve unrelated user edits and raw experimental evidence. Never commit data, models, caches, credentials, checkpoints or large outputs.
- Complete one logical unit with applicable checks and staged-diff/ignore review. Commit and push only when each operation is explicitly authorized. Verify remote SHA after a push and report failures. See CONTRIBUTING.md.
- Local `guiding/` is ignored and must never be staged, packaged or deployed. It may contain temporary mentoring instructions; it is not a runtime dependency or public documentation source.
