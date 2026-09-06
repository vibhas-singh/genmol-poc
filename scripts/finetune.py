#!/usr/bin/env python
"""Fine-tune a pretrained NVIDIA GenMol checkpoint on a focused SAFE dataset.

Approach
--------
1. Load `genmol/configs/base.yaml`, merge the PoC overrides (`poc/configs/finetune.yaml`),
   then apply any command-line overrides (data path, pretrained ckpt, steps, lr).
2. Instantiate `GenMol` from the pretrained checkpoint so both the backbone weights and
   the EMA shadow weights are inherited (`GenMol.load_from_checkpoint`).
3. Run a short Lightning `fit` on the user SAFE dataset with a fresh optimizer / step
   counter (we do NOT pass `ckpt_path`, so this is fine-tuning rather than resuming
   pretraining). A lower LR and short warmup adapt the model to the target domain.

Fine-tuned checkpoints are written to <out-dir>/checkpoints and consumed by
generate_evaluate.py.
"""

import argparse
import csv
import json
import os
import time

# Required for deterministic cuBLAS matmul; must be set before torch is imported.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from transformers import get_constant_schedule_with_warmup
from genmol.model import GenMol
from genmol.utils.ema import ExponentialMovingAverage
from genmol.utils.utils_data import Collator
from utils import plot_loss_curves

# Resolvers used inside base.yaml interpolations.
for name, fn in {
    "cwd": os.getcwd,
    "device_count": torch.cuda.device_count,
    "eval": eval,
    "div_up": lambda x, y: (x + y - 1) // y,
}.items():
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, fn)


class FreezeSafeEMA(ExponentialMovingAverage):
    """EMA that preserves every backbone tensor in checkpoint order, including frozen ones."""

    def __init__(self, parameters, decay):
        parameters = list(parameters)
        self.decay = decay
        self.num_updates = 0
        self.shadow_params = [param.clone().detach() for param in parameters]
        self.collected_params = []

    def update(self, parameters):
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        with torch.no_grad():
            for shadow, param in zip(self.shadow_params, parameters):
                shadow.sub_((1.0 - decay) * (shadow - param))

    def copy_to(self, parameters):
        for shadow, param in zip(self.shadow_params, parameters):
            param.data.copy_(shadow.data)


class SafeFileDataset(torch.utils.data.Dataset):
    """Map-style dataset that returns one complete SAFE record per index."""

    def __init__(self, data_path):
        with open(data_path) as data_file:
            self.safe_list = [line.rstrip("\n") for line in data_file]

    def __len__(self):
        return len(self.safe_list)

    def __getitem__(self, index):
        return {"input": self.safe_list[index]}


class FineTuneGenMol(GenMol):
    """GenMol subclass for fine-tuning (keeps the vendored repo untouched).

    Adds a validation step (val_loss for early stopping / best-checkpoint selection), a
    configurable warmup so a short run doesn't sit permanently inside the default 2500-step
    warmup, and EMA handling that lets a short fine-tune actually reflect in generation.
    """

    def on_load_checkpoint(self, checkpoint):
        # The sampler generates from EMA weights. The pretrained EMA carries decay=0.9999 and
        # ~50k num_updates, so over a short fine-tune the shadow barely moves and generation
        # keeps reflecting the pretrained distribution (looks like "no learning"). Reset the
        # averaging schedule and adopt the fine-tune decay so EMA tracks the adapted weights.
        super().on_load_checkpoint(checkpoint)
        if self.ema is not None:
            self.ema.decay = self.config.training.ema
            self.ema.num_updates = 0

    def configure_optimizers(self):
        # Only optimize params left trainable (all of them unless layers were frozen).
        params = [p for p in self.backbone.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config.optim.get("num_warmup_steps", 2500))
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": "lr"}]

    def on_validation_epoch_start(self):
        if self.ema is None:
            return
        self.ema.move_shadow_params_to_device(self.device)
        parameters = list(self.backbone.parameters())
        self.ema.store(parameters)
        self.ema.copy_to(parameters)

    def on_validation_epoch_end(self):
        if self.ema is not None:
            self.ema.restore(self.backbone.parameters())

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        losses = []
        for _ in range(self.config.validation.time_samples):
            t = self.mdlm.sample_time(input_ids.shape[0])
            xt = self.mdlm.forward_process(input_ids, t)
            with torch.amp.autocast("cuda", dtype=torch.float32):
                logits = self.backbone(xt, attention_mask)["logits"]
            if self.config.training.global_mean_loss:
                loss = self.mdlm.loss(
                    logits, input_ids, xt, t, mask=attention_mask, global_mean=True)
            else:
                loss = self.mdlm.loss(logits, input_ids, xt, t, mask=attention_mask).mean()
            losses.append(loss)
        loss = torch.stack(losses).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=input_ids.shape[0])
        return loss


def build_config(base_yaml, finetune_yaml, overrides):
    """Layer the configs: repo base.yaml <- PoC finetune.yaml <- CLI dotlist overrides."""
    config = OmegaConf.load(base_yaml)
    config = OmegaConf.merge(config, OmegaConf.load(finetune_yaml))
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    return config


def build_val_dataloader(config, data_path, val_data):
    """DataLoader over val.safe (for val_loss / early stopping), or None if it is absent."""
    val_path = val_data or os.path.join(os.path.dirname(data_path), "val.safe")
    if not os.path.exists(val_path):
        print(f"[warn] no val file at {val_path}: training without val_loss / early stopping.")
        return None
    return torch.utils.data.DataLoader(
        SafeFileDataset(val_path),
        batch_size=config.loader.batch_size,
        collate_fn=Collator(config),
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        shuffle=False)


def build_train_dataloader(config):
    """DataLoader over complete SAFE records from the user-provided training file."""
    return torch.utils.data.DataLoader(
        SafeFileDataset(config.data),
        batch_size=config.loader.batch_size,
        collate_fn=Collator(config),
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        shuffle=True,
        persistent_workers=config.loader.num_workers > 0)


def build_callbacks(ckpt_dir, every_n_epochs, has_val, patience):
    """Periodic per-epoch checkpoints; plus best-on-val checkpoint + early stopping when validating."""
    checkpoint = L.pytorch.callbacks.ModelCheckpoint
    callbacks = [checkpoint(
        dirpath=ckpt_dir, filename="{epoch}", save_top_k=-1,
        every_n_epochs=every_n_epochs,
        auto_insert_metric_name=False, enable_version_counter=False)]
    if has_val:
        callbacks.append(checkpoint(
            dirpath=ckpt_dir, filename="best", monitor="val_loss", mode="min",
            save_top_k=1, auto_insert_metric_name=False, enable_version_counter=False))
        if patience > 0:
            callbacks.append(L.pytorch.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=patience))
    return callbacks


def freeze_backbone_layers(model, num_frozen):
    """Freeze embeddings + the first `num_frozen` transformer layers (0 = full fine-tuning).

    A cheap regularizer for small datasets: keep the general-chemistry lower layers fixed and
    adapt only the upper layers + MLM head.
    """
    if num_frozen <= 0:
        return
    bert = model.backbone.bert
    n_layers = len(bert.encoder.layer)
    num_frozen = min(num_frozen, n_layers)
    for p in bert.embeddings.parameters():
        p.requires_grad = False
    for layer in bert.encoder.layer[:num_frozen]:
        for p in layer.parameters():
            p.requires_grad = False
    if model.ema is not None:
        model.ema = FreezeSafeEMA(
            model.backbone.parameters(), decay=model.config.training.ema)
        ema_shapes = [tuple(tensor.shape) for tensor in model.ema.shadow_params]
        parameter_shapes = [tuple(param.shape) for param in model.backbone.parameters()]
        if ema_shapes != parameter_shapes:
            raise RuntimeError("EMA shadows do not align with frozen-backbone parameter shapes")
        print("Reinitialized EMA over the full backbone to preserve checkpoint tensor alignment")
    trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.backbone.parameters())
    print(f"Froze embeddings + {num_frozen}/{n_layers} layers -> "
          f"trainable {trainable:,}/{total:,} ({100 * trainable / total:.1f}%)")


def write_run_records(out_dir, summary, registry_path):
    """Persist a per-run run_summary.json and append one row to the central runs_registry.csv."""
    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    write_header = not os.path.exists(registry_path)
    with open(registry_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary)
    print(f"Run summary -> {os.path.join(out_dir, 'run_summary.json')}")
    print(f"Registry    -> {registry_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-ckpt", required=True,
                        help="Path to pretrained checkpoint: model.ckpt (V1) or model_v2.ckpt (V2)")
    parser.add_argument("--data", required=True, help="Path to train.safe from prepare_data.py")
    parser.add_argument("--val-data", default=None,
                        help="Path to val.safe (default: val.safe next to --data). Enables "
                             "val_loss logging + early stopping.")
    parser.add_argument("--out-dir", required=True, help="Run directory for checkpoints/logs")
    parser.add_argument("--base-config", default=None, help="Path to genmol/configs/base.yaml")
    parser.add_argument("--finetune-config", default=None, help="Path to poc/configs/finetune.yaml")
    parser.add_argument("--use-bracket-safe", action=argparse.BooleanOptionalAction, default=True,
                        help="Fine-tune the V2 (extended/bracket SAFE) model [default]. Requires a "
                             "V2 checkpoint (model_v2.ckpt). The Collator converts SAFE->bracket-SAFE "
                             "on the fly, so the prepared data is unchanged. Pass --no-use-bracket-safe "
                             "to fine-tune the V1 model (model.ckpt) instead.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=25,
                        help="Early-stopping patience in epochs (0 disables early stop; the "
                             "default matches max_epochs so a run completes all 25 epochs).")
    parser.add_argument("--val-time-samples", type=int, default=None,
                        help="Independent diffusion-time samples averaged per validation batch "
                             "(default: validation.time_samples from config).")
    parser.add_argument("--freeze-layers", type=int, default=0,
                        help="Freeze embeddings + the first N transformer layers (0 = full "
                             "fine-tuning). A regularizer for small datasets; try 6-9 of 12.")
    parser.add_argument("--ema-decay", type=float, default=0.99,
                        help="EMA decay during fine-tuning. Generation samples from EMA weights, "
                             "and the pretraining default (0.9999) barely moves over a short run, "
                             "so the fine-tuned default is 0.99 for smoothing that still tracks "
                             "adaptation. Pass 0 to disable EMA and sample the backbone directly. "
                             "When layers are frozen, "
                             "EMA retains full-backbone checkpoint order while frozen values stay fixed.")
    parser.add_argument("--seed", type=int, default=42, help="Global seed for reproducibility.")
    parser.add_argument("--overrides", nargs="*", default=[], help="Extra dotlist overrides, e.g. loader.global_batch_size=256")
    args = parser.parse_args()

    # Seed Python/NumPy/torch (+ dataloader workers) so a run is as reproducible as possible.
    L.seed_everything(args.seed, workers=True)

    here = os.path.dirname(os.path.realpath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "genmol"))
    base_yaml = args.base_config or os.path.join(repo_root, "configs", "base.yaml")
    finetune_yaml = args.finetune_config or os.path.abspath(os.path.join(here, "..", "configs", "finetune.yaml"))

    # --- Config: base <- finetune <- CLI overrides (data path, steps, lr) ---
    # Checkpoint dir is passed directly to build_callbacks() below, not read from config.
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    overrides = list(args.overrides) + [f"data={args.data}"]
    if args.max_epochs is not None:
        overrides.append(f"trainer.max_epochs={args.max_epochs}")
    if args.lr is not None:
        overrides.append(f"optim.lr={args.lr}")
    if args.val_time_samples is not None:
        overrides.append(f"validation.time_samples={args.val_time_samples}")
    config = build_config(base_yaml, finetune_yaml, overrides)
    if config.validation.time_samples < 1:
        raise ValueError("validation.time_samples must be at least 1")

    # V2 (extended SAFE) uses angle-bracket tokens '<' '>': enable the flag and grow the
    # vocab by 2 so the architecture matches the V2 checkpoint (see scripts/train.py).
    if args.use_bracket_safe:
        config.training.use_bracket_safe = True
        config.model.vocab_size = config.model.vocab_size + 2

    # Use a faster EMA for fine-tuning; on_load_checkpoint resets its pretraining update history.
    # Passing 0 disables EMA so the sampler uses the fine-tuned backbone directly.
    config.training.ema = args.ema_decay

    os.makedirs(ckpt_dir, exist_ok=True)
    print(OmegaConf.to_yaml(config, resolve=False))

    if not os.path.exists(args.pretrained_ckpt):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {args.pretrained_ckpt}\n"
            "Download model_v2.ckpt (default) or model.ckpt (V1) from NGC: "
            "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/genmol_v1"
        )

    # --- Model + data: inherit backbone + EMA weights, then adapt on the target SAFE set ---
    model = FineTuneGenMol.load_from_checkpoint(args.pretrained_ckpt, config=config, strict=False)
    freeze_backbone_layers(model, args.freeze_layers)
    train_dataloader = build_train_dataloader(config)
    val_dataloader = build_val_dataloader(config, args.data, args.val_data)
    callbacks = build_callbacks(
        ckpt_dir, config.callback.get("every_n_epochs", 1), val_dataloader is not None, args.patience)

    # --- Train: fresh optimizer/step -> fine-tune (not resume). ckpt_path intentionally omitted. ---
    csv_logger = CSVLogger(save_dir=args.out_dir, name="metrics")
    trainer = L.Trainer(
        accelerator="cuda",
        devices=config.trainer.devices,
        num_nodes=1,
        precision=config.trainer.precision,
        max_epochs=config.trainer.max_epochs,
        gradient_clip_val=config.trainer.gradient_clip_val,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        log_every_n_steps=config.trainer.log_every_n_steps,
        check_val_every_n_epoch=config.trainer.get("check_val_every_n_epoch", 1),
        default_root_dir=args.out_dir,
        callbacks=callbacks,
        logger=csv_logger,
        deterministic="warn",  # deterministic kernels where available; warn (not error) otherwise
        enable_progress_bar=True,
    )
    # --- Baseline: full-val loss of the pretrained model before any training (step-0 anchor). ---
    baseline_val = None
    if val_dataloader is not None:
        baseline = trainer.validate(model, dataloaders=val_dataloader, verbose=False)
        baseline_val = float(baseline[0]["val_loss"])
        print(f"Baseline val_loss (pretrained, pre-finetune): {baseline_val:.4f}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)
    torch.cuda.synchronize()
    peak_gpu_allocated_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_gpu_reserved_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)
    print(f"Peak GPU memory during fine-tuning: allocated={peak_gpu_allocated_gib:.2f} GiB, "
          f"reserved={peak_gpu_reserved_gib:.2f} GiB")
    best_cb = next((c for c in callbacks
                    if isinstance(c, L.pytorch.callbacks.ModelCheckpoint) and c.monitor == "val_loss"), None)
    best_path = best_cb.best_model_path if best_cb else None
    best_val = (float(best_cb.best_model_score)
                if best_cb and best_cb.best_model_score is not None else None)
    plot_loss_curves(os.path.join(csv_logger.log_dir, "metrics.csv"),
                     args.out_dir, baseline_val)
    wall_time = time.time() - t0
    print(f"\nFine-tuning complete. Checkpoints in: {ckpt_dir}")

    # --- Run record: config + losses + early-stop step, saved per run and appended to a registry. ---
    early_cb = next((c for c in callbacks if isinstance(c, L.pytorch.callbacks.EarlyStopping)), None)
    stopped_early = bool(early_cb is not None and getattr(early_cb, "stopped_epoch", 0) > 0)
    if best_val is not None:
        print(f"Best val_loss: {best_val:.4f}")
        print(f"Best checkpoint: {best_path}")

    summary = {
        "out_dir": args.out_dir,
        "model_version": "v2" if config.training.get("use_bracket_safe") else "v1",
        "freeze_layers": args.freeze_layers,
        "ema_decay": float(config.training.ema),
        "lr": float(config.optim.lr),
        "seed": args.seed,
        "max_epochs": int(config.trainer.max_epochs),
        "patience": args.patience,
        "val_time_samples": int(config.validation.time_samples),
        "baseline_val_loss": baseline_val,
        "best_val_loss": best_val,
        "stopped_early": stopped_early,
        "stop_epoch": int(trainer.current_epoch),  # epoch training actually ended at
        "stop_step": int(trainer.global_step),     # total optimizer steps at stop
        "wall_time_sec": round(wall_time, 1),
        "peak_gpu_allocated_gib": round(peak_gpu_allocated_gib, 2),
        "peak_gpu_reserved_gib": round(peak_gpu_reserved_gib, 2),
        "best_ckpt": best_path,
    }
    registry_path = os.path.abspath(os.path.join(here, "..", "outputs", "runs_registry.csv"))
    write_run_records(args.out_dir, summary, registry_path)


if __name__ == "__main__":
    main()
