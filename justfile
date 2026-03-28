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

# Quick smoke test (1% data, 1 epoch)
test *ARGS:
    uv run python -m eval.run --config-name verify {{ARGS}}

# Verify all 9 models x 5 datasets (1% data, 1 epoch, frozen)
verify:
    #!/usr/bin/env bash
    set -uo pipefail
    models=(
      timm_resnet50
      timm_convnext_xlarge
      terratorch_dofa_large_patch16_224
      timm_clay_v1_base
      terratorch_satlas_swin_b_naip_si_rgb
      terratorch_satlas_swin_b_sentinel2_si_ms
      terratorch_ssl4eos12_resnet50_sentinel2_all_decur
      terratorch_terramind_v1_large
      terratorch_prithvi_eo_v2_600_tl
    )
    datasets=(spacenet2 spacenet7 flair2 dynamic_earthnet treesatai)
    log_dir="results/verify"; mkdir -p "$log_dir"
    pass=0; fail=0; total=$((${#models[@]} * ${#datasets[@]}))
    echo "Verifying $total model-dataset combinations (1% data, 1 epoch, frozen)..."
    for m in "${models[@]}"; do
      for d in "${datasets[@]}"; do
        printf "  %-55s x %-20s " "$m" "$d"
        log="$log_dir/${m}__${d}.log"
        if uv run python -m eval.run --config-name verify \
          model.backbone="$m" "data.datasets=[$d]" > "$log" 2>&1 \
          && ! grep -q "No results produced" "$log"; then
          echo "PASS"; ((pass++))
        else
          echo "FAIL (see $log)"; ((fail++))
        fi
      done
    done
    echo ""
    echo "$pass/$total passed, $fail/$total failed"

# Lint + format
lint:
    uv run ruff check --fix .
    uv run ruff format .

# Download GEO-Bench-2 VHR datasets
download:
    uv run geobench-download --root data spacenet2 spacenet7 flair2 dynamic_earthnet treesatai everwatch nzcattle
