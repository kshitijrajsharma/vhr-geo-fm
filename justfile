set dotenv-load

# Install dependencies
setup:
    uv sync

# Run with base config (pass hydra overrides)
run *ARGS:
    uv run python -m eval.run {{ARGS}}

# Frozen encoder evaluation (seg + cls datasets)
frozen *ARGS:
    uv run python -m eval.run --config-name frozen {{ARGS}}

# Full finetune evaluation (all datasets incl. detection)
finetune *ARGS:
    uv run python -m eval.run --config-name finetune {{ARGS}}

# Run pytest
test *ARGS:
    uv run pytest tests/ {{ARGS}}

# Quick smoke test (1% data, 1 epoch)
smoke *ARGS:
    uv run python -m eval.run --config-name verify {{ARGS}}

# Verify all 9 non-gated models x 5 datasets (sweep defined in verify.yaml)
verify:
    uv run python -m eval.run --config-name verify -m

# Lint + format
lint:
    uv run ruff check --fix .
    uv run ruff format .

# Type check
typecheck:
    uv run ty check

# CI check (no fixes, just check)
ci: lint typecheck test

# Download GEO-Bench-2 VHR datasets
download:
    uv run geobench-download --root data spacenet2 spacenet7 flair2 dynamic_earthnet treesatai everwatch nzcattle
