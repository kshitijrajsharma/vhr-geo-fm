"""GEO-Bench-2 VHR evaluation entry point.

Usage:
    uv run python -m eval.run                                     # defaults
    uv run python -m eval.run model.backbone=timm_resnet50        # override
    uv run python -m eval.run training.mode=finetune data.data_pct=5
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import socket
import tempfile
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import hydra
import lightning.pytorch as pl
import optuna
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig

from eval.datasets import DATASET_CONFIGS, create_datamodule
from eval.models import create_task
from eval.scoring import compute_scores, export_csv, print_table
from eval.subsample import SubsampledDataModule

warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

# PyTorch 2.6+ defaults weights_only=True, breaking terratorch/torchgeo checkpoints.
import lightning.fabric.plugins.io.torch_io as _torch_io  # noqa: E402

_orig_pl_load = _torch_io.pl_load
_torch_io.pl_load = lambda *a, **kw: _orig_pl_load(*a, **{**kw, "weights_only": False})  # ty: ignore[invalid-assignment]

optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger(__name__)


def _system_metadata() -> dict:
    meta = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "lightning_version": pl.__version__,
        "cpu_cores": os.cpu_count() or 0,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        meta |= {
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(total / (1024**3), 2),
            "gpu_free_gb": round(free / (1024**3), 2),
            "gpu_compute": f"{props.major}.{props.minor}",
        }
    return meta


def _resolve_workers(cfg: DictConfig) -> int:
    if cfg.performance.num_workers == -1:
        return max(2, (os.cpu_count() or 8) - 2)
    return max(0, int(cfg.performance.num_workers))


def _train_and_eval(
    cfg: DictConfig,
    dataset_name: str,
    lr: float,
    batch_size: int,
    seed: int,
    trial: optuna.Trial | None = None,
) -> tuple[float, float, float]:
    """Single train+eval run. Returns (val_metric, test_metric, seconds)."""
    config = DATASET_CONFIGS[dataset_name]
    pl.seed_everything(seed, workers=True)

    dm = create_datamodule(
        config,
        batch_size=batch_size,
        data_root=os.path.join(cfg.data.root, dataset_name),
        num_workers=_resolve_workers(cfg),
        pin_memory=bool(cfg.performance.pin_memory),
    )
    if cfg.data.data_pct < 100:
        dm = SubsampledDataModule(dm, data_pct=cfg.data.data_pct)

    task = create_task(
        config,
        backbone=cfg.model.backbone,
        lr=lr,
        weight_decay=cfg.training.weight_decay,
        frozen=cfg.training.mode == "frozen",
        pretrained=cfg.model.pretrained,
    )

    ckpt_dir = tempfile.mkdtemp(prefix=f"vhr_{dataset_name}_")
    callbacks: list[Callback] = [
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
            save_weights_only=True,
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

    t0 = time.perf_counter()
    trainer.fit(task, datamodule=dm)

    best_val = trainer.callback_metrics.get(config.metric_key)
    best_val = best_val.item() if best_val is not None else 0.0

    best_ckpt = os.path.join(ckpt_dir, "best.ckpt")
    ckpt_path = best_ckpt if os.path.exists(best_ckpt) else None
    try:
        test_results = trainer.test(task, datamodule=dm, ckpt_path=ckpt_path)
    except (ValueError, RuntimeError) as e:
        # PyTorch 2.6 / torchgeo Weights enum incompatibility during checkpoint unpickling
        log.warning("Checkpoint reload failed (%s), using last-epoch weights.", e)
        test_results = trainer.test(task, datamodule=dm, ckpt_path=None)

    test_key = config.metric_key.replace("val/", "test/").replace("val_", "test_")
    test_metric = test_results[0].get(test_key, 0.0) if test_results else 0.0

    shutil.rmtree(ckpt_dir, ignore_errors=True)
    return best_val, test_metric, time.perf_counter() - t0


def _run_hpo(cfg: DictConfig, dataset_name: str) -> dict:
    config = DATASET_CONFIGS[dataset_name]
    n_trials = cfg.hpo.n_trials
    print(f"\n{'=' * 60}\n  HPO: {dataset_name} ({n_trials} trials)\n{'=' * 60}")

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", cfg.hpo.lr_min, cfg.hpo.lr_max, log=True)
        bs = trial.suggest_categorical("batch_size", list(cfg.hpo.batch_sizes))
        try:
            val, _, _ = _train_and_eval(cfg, dataset_name, lr, bs, seed=42, trial=trial)
            return val
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


_DECODER_MAP = {
    "segmentation": "UNet",
    "classification": "IdentityDecoder",
    "detection": "faster-rcnn",
}


def _run_seeds(cfg: DictConfig, dataset_name: str, best_params: dict) -> list[dict]:
    config = DATASET_CONFIGS[dataset_name]
    seeds = list(cfg.eval.seeds)
    print(f"\n{'=' * 60}\n  Eval: {dataset_name} ({len(seeds)} seeds)\n{'=' * 60}")

    results = []
    for seed in seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)
        try:
            _, test_metric, secs = _train_and_eval(
                cfg,
                dataset_name,
                lr=best_params["lr"],
                batch_size=best_params["batch_size"],
                seed=seed,
            )
            print(f"test={test_metric:.6f} ({secs:.1f}s)")
            results.append(
                {
                    "dataset": dataset_name,
                    "metric_leaderboard": config.metric_leaderboard,
                    "test_metric": test_metric,
                    "seed": seed,
                    "batch_size": best_params["batch_size"],
                    "lr": best_params["lr"],
                    "decoder": _DECODER_MAP[config.task_type],
                    "early_stop_patience": cfg.training.early_stop_patience,
                    "n_trials": cfg.hpo.n_trials,
                    "weight_decay": best_params["weight_decay"],
                    "data_pct": cfg.data.data_pct,
                    "run_seconds": secs,
                }
            )
        except Exception as e:
            print(f"FAILED: {e}")
            log.exception("Seed %d failed for %s", seed, dataset_name)

    return results


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())

    t0 = time.perf_counter()
    started_at = datetime.now(UTC)
    meta = _system_metadata()
    frozen = cfg.training.mode == "frozen"

    print(
        f"GEO-Bench-2 VHR | {cfg.model.backbone} | {cfg.training.mode} | {cfg.data.data_pct}% data"
    )
    print(f"  Datasets: {list(cfg.data.datasets)}")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision(str(cfg.performance.matmul_precision))
        torch.backends.cudnn.benchmark = bool(cfg.performance.cudnn_benchmark)
        print(f"  GPU: {torch.cuda.get_device_name(0)} | workers={_resolve_workers(cfg)}")
    else:
        print("  WARNING: No GPU detected")

    dataset_names = list(cfg.data.datasets)
    if frozen:
        det = {"everwatch", "nzcattle"}
        skipped = [d for d in dataset_names if d in det]
        if skipped:
            print(f"  Skipping detection in frozen mode: {skipped}")
        dataset_names = [d for d in dataset_names if d not in det]

    all_results: list[dict] = []
    hpo_cache: dict[str, dict] = {}

    for ds in dataset_names:
        if ds not in DATASET_CONFIGS:
            print(f"  Unknown dataset: {ds}, skipping")
            continue
        if not (Path(cfg.data.root) / ds).exists():
            print(f"  Dataset {ds} not found, skipping")
            continue

        hpo_cache[ds] = _run_hpo(cfg, ds)
        all_results.extend(_run_seeds(cfg, ds, hpo_cache[ds]))

    if not all_results:
        print("No results produced.")
        return

    df = compute_scores(all_results)
    evaluated = [d for d in dataset_names if d in df["dataset"].unique().tolist()]
    print_table(df, evaluated)

    run_meta = {
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "total_run_seconds": round(time.perf_counter() - t0, 2),
    } | meta

    output_dir = Path(cfg.output.dir)
    export_csv(df, output_dir, cfg.model.backbone, frozen, run_metadata=run_meta)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hpo_params.json").write_text(json.dumps(hpo_cache, indent=2))
    print(f"\nResults -> {output_dir}")


if __name__ == "__main__":
    main()
