"""
Phase-1 Specialist Training Loop
==================================
Trains each specialist model independently:
  - CRNN2D  on heart-sound spectrograms
  - NTSNet  on ultrasound images
  - EfficientNetV2XRay  on chest X-rays

Uses:
  - BCEWithLogitsLoss
  - AdamW optimizer
  - StepLR scheduler
  - AMP (Automatic Mixed Precision)
  - TensorBoard logging

Run example:
  python training/train_specialist.py \
      --modality audio \
      --train_csv data/train.csv \
      --val_csv   data/val.csv   \
      --output_dir checkpoints
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import CFG
from data.dataset import AudioDataset, UltrasoundDataset, XRayDataset
from models.crnn_heart_sound import CRNN2D
from models.nts_net_ultrasound import NTSNet
from models.efficientnet_xray import EfficientNetV2XRay


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor):
    probs = torch.sigmoid(logits).cpu()
    preds = (probs >= 0.5).float()
    labels = labels.cpu()

    tp = ((preds == 1) & (labels == 1)).sum().float()
    tn = ((preds == 0) & (labels == 0)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    return {
        "accuracy": acc.item(),
        "sensitivity": sensitivity.item(),
        "specificity": specificity.item(),
    }


# ---------------------------------------------------------------------------
# Specialist model factory
# ---------------------------------------------------------------------------

def build_specialist(modality: str) -> nn.Module:
    if modality == "audio":
        return CRNN2D(
            embed_dim=CFG.audio.crnn_embed_dim,
            lstm_hidden=CFG.audio.lstm_hidden,
            lstm_layers=CFG.audio.lstm_layers,
            num_classes=1,
        )
    elif modality == "ultrasound":
        return NTSNet(
            embed_dim=CFG.ultrasound.nts_embed_dim,
            num_parts=CFG.ultrasound.nts_num_parts,
            top_k=CFG.ultrasound.nts_top_k,
            num_classes=1,
        )
    elif modality == "xray":
        return EfficientNetV2XRay(
            embed_dim=CFG.xray.efficientnet_embed_dim,
            num_classes=1,
        )
    else:
        raise ValueError(f"Unknown modality: {modality}. Choose from audio/ultrasound/xray")


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    modality: str,
    epoch: int,
    writer: SummaryWriter,
):
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for step, batch in enumerate(loader):
        label = batch["label"].unsqueeze(1).to(device)  # (B, 1)
        inp = batch["image"].to(device)  # unified key from single-modality datasets

        optimizer.zero_grad()
        with autocast(enabled=CFG.train.amp):
            logit = model(inp)          # (B, 1)
            loss = criterion(logit, label)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_logits.append(logit.detach().cpu())
        all_labels.append(label.detach().cpu())

        global_step = epoch * len(loader) + step
        if step % 20 == 0:
            writer.add_scalar(f"{modality}/train_loss_step", loss.item(), global_step)

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(
        torch.cat(all_logits).squeeze(1),
        torch.cat(all_labels).squeeze(1),
    )
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    modality: str,
):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for batch in loader:
        label = batch["label"].unsqueeze(1).to(device)
        inp = batch["image"].to(device)  # unified key from single-modality datasets

        with autocast(enabled=CFG.train.amp):
            logit = model(inp)
            loss = criterion(logit, label)

        total_loss += loss.item()
        all_logits.append(logit.cpu())
        all_labels.append(label.cpu())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(
        torch.cat(all_logits).squeeze(1),
        torch.cat(all_labels).squeeze(1),
    )
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_specialist(
    modality: str,
    train_csv: str,
    val_csv: str,
    output_dir: str = "checkpoints",
    resume: str | None = None,
):
    set_seed(CFG.train.seed)
    device = torch.device(CFG.train.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Modality: {modality}")

    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(CFG.train.log_dir, modality))

    # ── Datasets (per-modality) ──
    _DS = {"audio": AudioDataset, "ultrasound": UltrasoundDataset, "xray": XRayDataset}
    DatasetCls = _DS[modality]
    train_ds = DatasetCls(train_csv, augment=True)
    val_ds = DatasetCls(val_csv, augment=False)
    print(f"  Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.train.specialist_batch_size,
        shuffle=True,
        num_workers=CFG.train.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.train.specialist_batch_size,
        shuffle=False,
        num_workers=CFG.train.num_workers,
        pin_memory=True,
    )

    # ── Model ──
    model = build_specialist(modality).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Optimiser & Scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.train.specialist_lr,
        weight_decay=CFG.train.specialist_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CFG.train.scheduler_step_size,
        gamma=CFG.train.scheduler_gamma,
    )
    # Positive-weight for class imbalance (assume ~20% positive)
    pos_weight = torch.tensor([4.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = GradScaler(enabled=CFG.train.amp)

    # ── Optional resume ──
    start_epoch = 0
    best_acc = 0.0
    if resume is not None:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # ── Training Loop ──
    for epoch in range(start_epoch, CFG.train.specialist_epochs):
        t0 = time.time()
        tr_loss, tr_met = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, modality, epoch, writer
        )
        val_loss, val_met = validate(model, val_loader, criterion, device, modality)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:03d}/{CFG.train.specialist_epochs} | "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_met['accuracy']:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_met['accuracy']:.3f} "
            f"Sen: {val_met['sensitivity']:.3f} Spe: {val_met['specificity']:.3f} | "
            f"{elapsed:.1f}s"
        )

        # TensorBoard
        writer.add_scalar(f"{modality}/train_loss", tr_loss, epoch)
        writer.add_scalar(f"{modality}/val_loss", val_loss, epoch)
        for k, v in val_met.items():
            writer.add_scalar(f"{modality}/val_{k}", v, epoch)

        # Checkpoint
        is_best = val_met["accuracy"] > best_acc
        if is_best:
            best_acc = val_met["accuracy"]

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_metrics": val_met,
            "best_acc": best_acc,
        }
        ckpt_path = os.path.join(output_dir, f"{modality}_epoch{epoch+1:03d}.pth")
        torch.save(ckpt, ckpt_path)

        if is_best:
            best_path = os.path.join(output_dir, f"{modality}_best.pth")
            torch.save(ckpt, best_path)
            print(f"  ✓ New best saved: acc={best_acc:.3f}")

        if best_acc >= 0.85:
            print(f"  ✓ Target accuracy 85% reached. Phase-1 training complete.")

    writer.close()
    print(f"Training complete. Best val accuracy: {best_acc:.3f}")
    return best_acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase-1 Specialist Training")
    parser.add_argument("--modality", choices=["audio", "ultrasound", "xray"], required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    train_specialist(
        modality=args.modality,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        resume=args.resume,
    )
