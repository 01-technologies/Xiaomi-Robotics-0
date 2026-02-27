# AGENTS.md

## Purpose
- Guidance for agentic coding tools operating in this repository.
- Scope: full repo (`deploy/`, `eval_calvin/`, `eval_libero/`, `eval_simplerenv/`, `scripts/`, root config).
- Prefer minimal, targeted edits; preserve backward-compatible CLI behavior.

## External Rule Files
- Checked `.cursor/rules/`: not found.
- Checked `.cursorrules`: not found.
- Checked `.github/copilot-instructions.md`: not found.
- If these files are added later, treat them as higher-priority constraints and update this file.

## Repo Snapshot
- Language: Python.
- Runtime service path: `deploy/server.py` + `deploy/client.py`.
- Real robot adapter: `deploy/piper_adapter.py`.
- Benchmark entrypoints:
  - LIBERO: `eval_libero/main.py`, `eval_libero/merge_results.py`
  - CALVIN: `eval_calvin/main.py`, `eval_calvin/merge_results.py`
  - SimplerEnv: `eval_simplerenv/main.py`
- Orchestration scripts: `scripts/deploy.sh`, `scripts/launch_*.sh`.

## Environment Notes
- Root project (`pyproject.toml`) targets Python `>=3.12,<3.13`.
- Benchmark READMEs currently use separate conda envs with Python 3.10.
- Script env names are fixed:
  - `scripts/deploy.sh` -> `mibot`
  - `scripts/launch_calvin.sh` -> `calvin`
  - `scripts/launch_libero.sh` -> `libero`
  - `scripts/launch_simplerenv.sh` -> `simplerenv`
- `scripts/launch_piper_stack.sh` is uv-first: use `uv run` (do not add conda activation back).
- `scripts/launch_piper_stack.sh` defaults to OpenCV camera mode (`--base-cam-index 0`) and PiPER SDK control.
- Treat ROS camera mode as optional fallback in PiPER flows (target runtime often does not have ROS installed).
- For AMD/ROCm systems, run `source set_envs.sh` before starting model servers.

## Build / Setup
```bash
# Preferred dependency install
uv sync
# Fallback editable install
pip install -e .
# Optional ROCm env vars
source set_envs.sh
# Start one server
python deploy/server.py --model XiaomiRobotics/Xiaomi-Robotics-0-LIBERO --port 10086
# Start multi-port servers via tmux
bash scripts/deploy.sh XiaomiRobotics/Xiaomi-Robotics-0-LIBERO 8 8
```

## Lint / Format
- No enforced repo-level lint config is present in `pyproject.toml` yet.
```bash
# Lint (recommended)
uv run ruff check .
# Format
uv run ruff format .
# Syntax-only fallback
python -m compileall deploy eval_calvin eval_libero eval_simplerenv
# Install tooling if missing
uv pip install ruff
```

## Tests
- Current state: no first-party `tests/` directory.
- If pytest tests are added, use:
```bash
# All tests
uv run pytest -q
# Single test function (important)
uv run pytest path/to/test_file.py::test_name -q
# Single test file
uv run pytest path/to/test_file.py -q
# Keyword subset
uv run pytest -k "keyword" -q
```

## Integration Smoke Commands (Single-Task Equivalents)
```bash
# LIBERO: one task, one trial
python eval_libero/main.py \
  --args.task-suite-name libero_10 \
  --args.task-id 0 \
  --args.num-trials-per-task 1 \
  --args.num-workers 1 \
  --args.video-out-path ./logs/libero_smoke \
  --args.port 10086
# CALVIN: one sequence on one rank
python eval_calvin/main.py \
  --rank 0 \
  --world_size 1 \
  --num-sequences 1 \
  --num-workers 1 \
  --CACHE_ROOT ./logs/calvin_smoke \
  --dataset_path /path/to/Calvin/task_ABCD_D
# Merge CALVIN outputs
python eval_calvin/merge_results.py --eval_log_dir ./logs/calvin_smoke
# SimplerEnv: one worker minimal run
python eval_simplerenv/main.py \
  --args.dataset-name fractal \
  --args.worker-id 0 \
  --args.num-workers 1 \
  --args.repeat-times 1 \
  --args.experiment-root ./logs/simplerenv_smoke \
  --args.port 10086
```

## Code Style Guidelines

### Imports
- Group imports as: standard library, third-party, local modules.
- Separate groups with one blank line.
- Keep imports explicit; avoid wildcard imports.
- Remove unused imports in touched files.
- Avoid adding new `sys.path.append(...)` unless script-entry constraints require it.

### Formatting
- Use 4-space indentation and UTF-8 files.
- Target readable lines (~100-120 chars unless the file already uses longer lines).
- Use trailing commas in multiline calls/literals.
- Prefer one logical statement per line.
- Add comments only where behavior is non-obvious.

### Types and APIs
- Add type hints for new/modified function signatures.
- Prefer concrete annotations (`np.ndarray`, `Path`, `Literal[...]`, `dict[str, Any]`).
- Document tensor/array shape expectations in model IO docstrings.
- Preserve dtype/device behavior in inference code (especially bfloat16 paths).
- Avoid silent dtype conversions unless required by external APIs.

### Naming and CLI
- Functions/variables: `snake_case`; classes: `PascalCase`; constants: `UPPER_SNAKE_CASE`.
- Keep CLI dataclasses named `Args` for consistency with current scripts.
- Prefer dataclass-based CLIs with `tyro` for new evaluation flows.
- Do not remove flags consumed by `scripts/launch_*.sh`.
- When adding options, update relevant README usage snippets.

### Errors, Logging, and Safety
- Raise specific exceptions with actionable messages (`ValueError`, `RuntimeError`, `ConnectionError`).
- Catch broad exceptions only at process boundaries; log context.
- Use `logging` in long-running loops; reserve `print` for concise user-facing status.
- Preserve cleanup via `finally`, `close()`, and environment shutdown paths.
- Never swallow exceptions silently.

### Concurrency, Paths, and Reproducibility
- Guard shared mutable state with locks/managers in multiprocessing code.
- Serialize shared model-client calls when socket/resource contention is possible.
- Prefer `pathlib.Path`; create output dirs with `mkdir(parents=True, exist_ok=True)`.
- Keep generated artifacts under `./logs` or benchmark-specific output roots.
- Preserve deterministic seed derivation behavior (`hash_data_to_seed`) and explicit seeding.

## Agent Checklist
- Make the smallest safe change that solves the task.
- Run relevant lint/syntax checks on changed files.
- For benchmark-logic changes, run at least one single-task smoke command.
- Update README/help text when CLI behavior changes.
- Do not commit generated logs/videos/results artifacts.
