"""
Phase-2 GMU Training Loop
===========================
Loads the three trained specialist encoders (surgery-mode),
freezes them, and trains ONLY the GMU + MLP weights.

Run example:
  python training/train_gmu.py \
      --audio_ckpt    checkpoints/audio_best.pth    \
      --us_ckpt       checkpoints/ultrasound_best.pth \
      --xray_ckpt     checkpoints/xray_best.pth      \
      --train_csv     data/train.csv                 \
      --val_csv       data/val.csv                   \
      --output_dir    checkpoints
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import CFG
from data.dataset import PediatricCardiacDataset, collate_fn
from models.crnn_heart_sound import CRNN2D, crnn_without_head
from models.nts_net_ultrasound import NTSNet, ntsnet_without_head
from models.efficientnet_xray import EfficientNetV2XRay, efficientnet_without_head
from models.gmu_fusion import MultimodalModel


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return {
        "accuracy": acc.item(),
        "sensitivity": sensitivity.item(),
        "specificity": specificity.item(),
        "f1": f1.item(),
    }


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: MultimodalModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
):
    model.train()
    # Keep specialist encoders in eval mode (frozen BN stats)
    model.crnn.eval()
    model.nts_net.eval()
    model.effnet.eval()

    total_loss = 0.0
    all_logits, all_labels = [], []

    for step, batch in enumerate(loader):
        audio_spec = batch["audio_spec"].to(device)
        us_image = batch["us_image"].to(device)
        xray_image = batch["xray_image"].to(device)
        label = batch["label"].unsqueeze(1).to(device)

        optimizer.zero_grad()
        with autocast(enabled=CFG.train.amp):
            logit, gate_weights = model(audio_spec, us_image, xray_image)
            loss = criterion(logit, label)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=5.0,
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_logits.append(logit.detach().cpu())
        all_labels.append(label.detach().cpu())

        global_step = epoch * len(loader) + step
        if step % 10 == 0:
            writer.add_scalar("gmu/train_loss_step", loss.item(), global_step)
            # Log mean gate weights for interpretability
            for mod_name, w in gate_weights.items():
                writer.add_scalar(
                    f"gates/{mod_name}_mean", w.mean().item(), global_step
                )

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
    model: MultimodalModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for batch in loader:
        audio_spec = batch["audio_spec"].to(device)
        us_image = batch["us_image"].to(device)
        xray_image = batch["xray_image"].to(device)
        label = batch["label"].unsqueeze(1).to(device)

        with autocast(enabled=CFG.train.amp):
            logit, _ = model(audio_spec, us_image, xray_image)
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
# Main
# ---------------------------------------------------------------------------

def train_gmu(
    audio_ckpt: str,
    us_ckpt: str,
    xray_ckpt: str,
    train_csv: str,
    val_csv: str,
    output_dir: str = "checkpoints",
    resume_gmu: str | None = None,
):
    set_seed(CFG.train.seed)
    device = torch.device(CFG.train.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(CFG.train.log_dir, "gmu"))

    # ── Load Specialists (surgery) ──
    print("Loading specialist encoders (surgery mode)...")
    crnn = crnn_without_head(audio_ckpt, device=str(device))
    nts = ntsnet_without_head(us_ckpt, device=str(device))
    eff = efficientnet_without_head(xray_ckpt, device=str(device))

    # ── Build Multimodal Model ──
    model = MultimodalModel(
        crnn_model=crnn,
        nts_model=nts,
        effnet_model=eff,
        audio_dim=CFG.fusion.audio_embed_dim,
        us_dim=CFG.fusion.us_embed_dim,
        xray_dim=CFG.fusion.xray_embed_dim,
        gmu_hidden=CFG.fusion.gmu_hidden_dim,
        mlp_hidden=CFG.fusion.mlp_hidden_dims,
        mlp_dropout=CFG.fusion.mlp_dropout,
        modality_drop_p=CFG.fusion.modality_drop_p,
        freeze_specialists=True,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    # ── Datasets ──
    train_ds = PediatricCardiacDataset(train_csv, augment=True)
    val_ds = PediatricCardiacDataset(val_csv, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.train.gmu_batch_size,
        shuffle=True,
        num_workers=CFG.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.train.gmu_batch_size,
        shuffle=False,
        num_workers=CFG.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ── Optimiser — only GMU + MLP parameters ──
    gmu_params = list(model.gmu.parameters()) + list(model.mlp.parameters())
    optimizer = torch.optim.AdamW(
        gmu_params,
        lr=CFG.train.gmu_lr,
        weight_decay=CFG.train.gmu_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.train.gmu_epochs, eta_min=1e-6
    )
    pos_weight = torch.tensor([4.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = GradScaler(enabled=CFG.train.amp)

    # ── Optional resume ──
    start_epoch = 0
    best_f1 = 0.0
    if resume_gmu is not None:
        ckpt = torch.load(resume_gmu, map_location=device)
        model.gmu.load_state_dict(ckpt["gmu_state_dict"])
        model.mlp.load_state_dict(ckpt["mlp_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # ── Training Loop ──
    for epoch in range(start_epoch, CFG.train.gmu_epochs):
        t0 = time.time()
        tr_loss, tr_met = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch, writer
        )
        val_loss, val_met = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:02d}/{CFG.train.gmu_epochs} | "
            f"Train Loss: {tr_loss:.4f} F1: {tr_met['f1']:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_met['accuracy']:.3f} "
            f"F1: {val_met['f1']:.3f} Sen: {val_met['sensitivity']:.3f} "
            f"Spe: {val_met['specificity']:.3f} | {elapsed:.1f}s"
        )

        for k, v in val_met.items():
            writer.add_scalar(f"gmu/val_{k}", v, epoch)
        writer.add_scalar("gmu/val_loss", val_loss, epoch)
        writer.add_scalar("gmu/train_loss", tr_loss, epoch)

        is_best = val_met["f1"] > best_f1
        if is_best:
            best_f1 = val_met["f1"]

        ckpt = {
            "epoch": epoch,
            "gmu_state_dict": model.gmu.state_dict(),
            "mlp_state_dict": model.mlp.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_met,
            "best_f1": best_f1,
        }
        torch.save(ckpt, os.path.join(output_dir, f"gmu_epoch{epoch+1:02d}.pth"))
        if is_best:
            torch.save(ckpt, os.path.join(output_dir, "gmu_best.pth"))
            print(f"  ✓ New best GMU saved: F1={best_f1:.3f}")

    writer.close()
    print(f"GMU training complete. Best F1: {best_f1:.3f}")
    return best_f1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase-2 GMU Training")
    parser.add_argument("--audio_ckpt", required=True)
    parser.add_argument("--us_ckpt", required=True)
    parser.add_argument("--xray_ckpt", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--resume_gmu", default=None)
    args = parser.parse_args()

    train_gmu(
        audio_ckpt=args.audio_ckpt,
        us_ckpt=args.us_ckpt,
        xray_ckpt=args.xray_ckpt,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        resume_gmu=args.resume_gmu,
    )
