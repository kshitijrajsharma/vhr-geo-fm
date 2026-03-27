# VHR GEO-FM Evaluation Framework

# Install all dependencies
setup:
    uv sync

# Run evaluation with hydra overrides
run *ARGS:
    uv run python -m eval.run {{ARGS}}

# Fix all formatting , lint & type issues
lint:
    uv run ruff check --fix .
    uv run ruff format .
    uv run ty check .

# Download all 7 VHR datasets
download:
    uv run geobench-download --root data spacenet2 spacenet7 flair2 dynamic_earthnet treesatai everwatch nzcattle

# Smoke test: ResNet50 (25M), 1 trial, 1 seed, 2 epochs on spacenet2
test:
    uv run python -m eval.run model.backbone=timm_resnet50 'data.datasets=[spacenet2]' hpo.n_trials=1 'eval.seeds=[42]' training.max_epochs=2 training.mode=frozen

# Full validation: reproduce ResNet50 fine-tuned on SpaceNet2 (~52.8 normalized)
validate:
    uv run python -m eval.run model.backbone=timm_resnet50 'data.datasets=[spacenet2]' training.mode=finetune
