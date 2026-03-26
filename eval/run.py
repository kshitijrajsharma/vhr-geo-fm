"""GEO-Bench-2 VHR evaluation entry point.

Usage:
    uv run python -m eval.run                                     # defaults
    uv run python -m eval.run model.backbone=timm_resnet50        # override model
    uv run python -m eval.run training.mode=finetune data.data_pct=5
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import hydra
import lightning.pytorch as pl
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig

from eval.datasets import DATASET_CONFIGS, create_datamodule
from eval.models import create_task
from eval.scoring import compute_scores, export_csv, export_model_info, print_table
from eval.subsample import SubsampledDataModule

warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)

log = logging.getLogger(__name__)


def _train_and_eval(
    cfg: DictConfig,
    dataset_name: str,
    lr: float,
    batch_size: int,
    seed: int,
    trial: optuna.Trial | None = None,
) -> tuple[float, float]:
    """Train and evaluate a single run. Returns (val_metric, test_metric)."""
    config = DATASET_CONFIGS[dataset_name]
    pl.seed_everything(seed, workers=True)

    if cfg.performance.num_workers == -1:
        available_cores = os.cpu_count() or 8
        num_workers = max(2, available_cores - 2)
    else:
        num_workers = max(0, int(cfg.performance.num_workers))

    dm = create_datamodule(
        config,
        batch_size=batch_size,
        data_root=os.path.join(cfg.data.root, dataset_name),
        num_workers=num_workers,
        pin_memory=bool(cfg.performance.pin_memory),
    )
    if cfg.data.data_pct < 100:
        dm = SubsampledDataModule(dm, data_pct=cfg.data.data_pct)

    frozen = cfg.training.mode == "frozen"
    task = create_task(
        config,
        backbone=cfg.model.backbone,
        lr=lr,
        weight_decay=cfg.training.weight_decay,
        frozen=frozen,
        pretrained=cfg.model.pretrained,
    )

    ckpt_dir = tempfile.mkdtemp(prefix=f"vhr_{dataset_name}_")
    callbacks = [
        EarlyStopping(
            monitor=config.metric_key,
            patience=cfg.training.early_stop_patience,
            mode=config.metric_direction,
            verbose=False,
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor=config.metric_key,
            mode=config.metric_direction,
            save_top_k=1,
            filename="best",
        ),
    ]

    if trial is not None:
        from optuna.integration import PyTorchLightningPruningCallback

        callbacks.append(PyTorchLightningPruningCallback(trial, monitor=config.metric_key))

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision=cfg.training.precision,
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        benchmark=bool(cfg.performance.cudnn_benchmark),
        deterministic=False,
    )

    trainer.fit(task, datamodule=dm)

    best_val = trainer.callback_metrics.get(config.metric_key)
    best_val = best_val.item() if best_val is not None else 0.0

    best_ckpt = os.path.join(ckpt_dir, "best.ckpt")
    ckpt_path = best_ckpt if os.path.exists(best_ckpt) else None
    test_results = trainer.test(task, datamodule=dm, ckpt_path=ckpt_path)

    test_key = config.metric_key.replace("val/", "test/").replace("val_", "test_")
    test_metric = test_results[0].get(test_key, 0.0) if test_results else 0.0

    shutil.rmtree(ckpt_dir, ignore_errors=True)
    return best_val, test_metric


def _run_hpo(cfg: DictConfig, dataset_name: str) -> dict:
    """Run HPO with Optuna to find best lr and batch_size."""
    config = DATASET_CONFIGS[dataset_name]
    n_trials = cfg.hpo.n_trials

    print(f"\n{'=' * 60}")
    print(f"  HPO for {dataset_name} ({n_trials} trials)")
    print(f"{'=' * 60}")

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", cfg.hpo.lr_min, cfg.hpo.lr_max, log=True)
        batch_size = trial.suggest_categorical("batch_size", list(cfg.hpo.batch_sizes))
        try:
            val_metric, _ = _train_and_eval(cfg, dataset_name, lr, batch_size, seed=42, trial=trial)
            return val_metric
        except Exception as e:
            log.warning("Trial failed: %s", e)
            return float("-inf") if config.metric_direction == "max" else float("inf")

    direction = "maximize" if config.metric_direction == "max" else "minimize"
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["weight_decay"] = cfg.training.weight_decay
    print(f"  Best: lr={best['lr']:.6f}, bs={best['batch_size']}, val={study.best_value:.6f}")
    return best


def _run_seeds(cfg: DictConfig, dataset_name: str, best_params: dict) -> list[dict]:
    """Run repeated evaluation with best HPO params across all seeds."""
    config = DATASET_CONFIGS[dataset_name]
    seeds = list(cfg.eval.seeds)

    print(f"\n{'=' * 60}")
    print(f"  Repeated evaluation for {dataset_name} ({len(seeds)} seeds)")
    print(f"{'=' * 60}")

    results = []
    for seed in seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)
        try:
            _, test_metric = _train_and_eval(
                cfg,
                dataset_name,
                lr=best_params["lr"],
                batch_size=best_params["batch_size"],
                seed=seed,
            )
            print(f"test={test_metric:.6f}")
            decoder = (
                "UNet"
                if config.task_type == "segmentation"
                else ("IdentityDecoder" if config.task_type == "classification" else "faster-rcnn")
            )
            results.append(
                {
                    "dataset": dataset_name,
                    "metric_leaderboard": config.metric_leaderboard,
                    "test_metric": test_metric,
                    "seed": seed,
                    "batch_size": best_params["batch_size"],
                    "lr": best_params["lr"],
                    "decoder": decoder,
                    "early_stop_patience": cfg.training.early_stop_patience,
                    "n_trials": cfg.hpo.n_trials,
                    "weight_decay": best_params["weight_decay"],
                    "data_pct": cfg.data.data_pct,
                }
            )
        except Exception as e:
            print(f"FAILED: {e}")
            log.exception("Seed %d failed for %s", seed, dataset_name)

    return results


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Hydra changes CWD; resolve paths relative to original working dir
    orig_cwd = hydra.utils.get_original_cwd()
    os.chdir(orig_cwd)

    print("GEO-Bench-2 VHR Evaluation")
    print(f"  Backbone: {cfg.model.backbone}")
    print(f"  Mode: {cfg.training.mode}")
    print(f"  Data %: {cfg.data.data_pct}")
    print(f"  Datasets: {list(cfg.data.datasets)}")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision(str(cfg.performance.matmul_precision))
        torch.backends.cudnn.benchmark = bool(cfg.performance.cudnn_benchmark)
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: No GPU detected")

    if cfg.performance.num_workers == -1:
        available_cores = os.cpu_count() or 8
        effective_workers = max(2, available_cores - 2)
    else:
        effective_workers = max(0, int(cfg.performance.num_workers))

    print(
        "  Perf: workers="
        f"{effective_workers}, pin_memory={cfg.performance.pin_memory}, "
        f"matmul_precision={cfg.performance.matmul_precision}, "
        f"cudnn_benchmark={cfg.performance.cudnn_benchmark}"
    )

    frozen = cfg.training.mode == "frozen"
    dataset_names = list(cfg.data.datasets)

    # Skip detection datasets in frozen mode (not available per GEO-Bench-2)
    if frozen:
        detection_ds = {"everwatch", "nzcattle"}
        skipped = [d for d in dataset_names if d in detection_ds]
        if skipped:
            print(f"  Skipping detection datasets in frozen mode: {skipped}")
        dataset_names = [d for d in dataset_names if d not in detection_ds]

    all_results = []
    hpo_cache = {}

    for ds_name in dataset_names:
        if ds_name not in DATASET_CONFIGS:
            print(f"  Unknown dataset: {ds_name}, skipping")
            continue

        ds_path = Path(cfg.data.root) / ds_name
        if not ds_path.exists():
            print(f"  Dataset {ds_name} not found at {ds_path}, skipping")
            continue

        best_params = _run_hpo(cfg, ds_name)
        hpo_cache[ds_name] = best_params

        results = _run_seeds(cfg, ds_name, best_params)
        all_results.extend(results)

    if not all_results:
        print("No results produced.")
        return

    df = compute_scores(all_results)
    evaluated_datasets = [d for d in dataset_names if d in df["dataset"].unique().tolist()]
    print_table(df, evaluated_datasets)

    output_dir = Path(cfg.output.dir)
    export_csv(df, output_dir, cfg.model.backbone, frozen)

    model_registry_path = Path("models.json")
    if model_registry_path.exists():
        with open(model_registry_path) as f:
            model_registry = json.load(f)
        export_model_info(output_dir, cfg.model.backbone, model_registry)

    # Save HPO params
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "hpo_params.json", "w") as f:
        json.dump(hpo_cache, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
