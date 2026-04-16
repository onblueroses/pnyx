#!/usr/bin/env python3
# pyright: reportCallIssue=false, reportArgumentType=false
"""
Deliberation Quality Model - DeBERTa-v3-small with multi-head DQI output.

Adapted from unslop v0.5 training pipeline. Multi-head classification:
  - justification: 4 classes (0-3)
  - respect: 4 classes (0-3)
  - constructiveness: 3 classes (0-2)

Labels of -1 are masked (ignored in loss computation).

Usage:
    python train_deliberation.py --data data/training/unified.jsonl
    python train_deliberation.py --data data/training/unified.jsonl --synthetic data/training/synthetic_dqi.jsonl
    python train_deliberation.py --dry-run
"""

import argparse
import hashlib
import json
import math
import os

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "microsoft/deberta-v3-small"
HIDDEN_SIZE = 768
MAX_SEQ_LEN = 256
HEAD_SIZES = {"justification": 4, "respect": 4, "constructiveness": 3}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", required=True, help="Path to unified.jsonl")
    p.add_argument(
        "--synthetic", default=None, help="Path to synthetic_dqi.jsonl (optional)"
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--out-dir", default=os.path.expanduser("~/deliberation-experiment"))
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--export-onnx", action="store_true")
    return p.parse_args()


def load_data(path, synthetic_path=None):
    """Load JSONL training data."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if synthetic_path and os.path.exists(synthetic_path):
        with open(synthetic_path, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def main():
    args = parse_args()

    config = {
        "model": MODEL_NAME,
        "hidden_size": HIDDEN_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "heads": HEAD_SIZES,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
    }
    print(f"Config: {json.dumps(config, indent=2)}")

    # Load data early to validate
    all_data = load_data(args.data, args.synthetic)
    print(f"Total samples: {len(all_data)}")

    for dim, n_classes in HEAD_SIZES.items():
        labeled = [r for r in all_data if r.get(dim, -1) >= 0]
        dist = {}
        for r in labeled:
            v = r[dim]
            dist[v] = dist.get(v, 0) + 1
        print(f"  {dim}: {len(labeled)} labeled, dist={dict(sorted(dist.items()))}")

    if args.dry_run:
        print("Dry run - exiting.")
        return

    # ── ML imports (deferred for dry-run support) ────────────────────────────

    import time

    import torch  # type: ignore[import-not-found]
    import torch.nn as nn  # type: ignore[import-not-found]
    from torch.utils.data import DataLoader, Dataset  # type: ignore[import-not-found]
    from torch.cuda.amp import autocast, GradScaler  # type: ignore[import-not-found]
    from transformers import AutoTokenizer, AutoModel  # type: ignore[import-not-found]
    from sklearn.metrics import f1_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}")
    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}  "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Train/val split ──────────────────────────────────────────────────────

    # Hash-based split: deterministic regardless of load order
    val_data = [
        r
        for r in all_data
        if hashlib.sha256(r["text"].strip().encode()).hexdigest() >= "cc"
    ]
    train_data = [
        r
        for r in all_data
        if hashlib.sha256(r["text"].strip().encode()).hexdigest() < "cc"
    ]
    print(f"Train: {len(train_data)}  Val: {len(val_data)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Dataset ──────────────────────────────────────────────────────────────

    class DQIDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            text = item["text"]
            enc = tokenizer(
                text,
                max_length=MAX_SEQ_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            labels = {
                dim: torch.tensor(item.get(dim, -1), dtype=torch.long)
                for dim in HEAD_SIZES
            }
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                **labels,
            }

    train_loader = DataLoader(
        DQIDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        DQIDataset(val_data),
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────

    class DeliberationScorer(nn.Module):
        """DeBERTa backbone + per-dimension classification heads."""

        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.norm = nn.LayerNorm(HIDDEN_SIZE)
            self.dropout = nn.Dropout(0.1)

            # One head per DQI dimension
            self.heads = nn.ModuleDict(
                {
                    dim: nn.Sequential(
                        nn.Linear(HIDDEN_SIZE, 256),
                        nn.GELU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, n_classes),
                    )
                    for dim, n_classes in HEAD_SIZES.items()
                }
            )

        def forward(self, input_ids, attention_mask):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            cls = self.dropout(self.norm(cls))
            return {dim: head(cls) for dim, head in self.heads.items()}

    backbone = AutoModel.from_pretrained(MODEL_NAME)

    # Freeze embeddings only
    for param in backbone.embeddings.parameters():
        param.requires_grad = False

    model = DeliberationScorer(backbone).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {trainable:,} trainable / {total_params:,} total")

    # ── Optimizer + scheduler ────────────────────────────────────────────────

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criteria = {dim: nn.CrossEntropyLoss(ignore_index=-1) for dim in HEAD_SIZES}
    scaler = GradScaler(enabled=use_amp)

    print(f"\nTraining: {args.epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"Schedule: {warmup_steps} warmup / {total_steps} total steps\n")

    # ── Training loop ────────────────────────────────────────────────────────

    best_f1 = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast(enabled=use_amp):
                logits = model(input_ids, attention_mask)
                loss = sum(
                    criteria[dim](logits[dim], batch[dim].to(device))
                    for dim in HEAD_SIZES
                ) / len(HEAD_SIZES)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss += loss.item()  # type: ignore[union-attr]
            global_step += 1

            if global_step % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  step {global_step}: loss={loss.item():.4f} lr={lr:.2e}")  # type: ignore[union-attr]

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        # ── Validation ───────────────────────────────────────────────────────

        model.eval()
        all_preds = {dim: [] for dim in HEAD_SIZES}
        all_labels = {dim: [] for dim in HEAD_SIZES}

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with autocast(enabled=use_amp):
                    logits = model(input_ids, attention_mask)

                for dim in HEAD_SIZES:
                    labels = batch[dim]
                    preds = logits[dim].argmax(dim=-1).cpu()
                    mask = labels >= 0
                    all_preds[dim].extend(preds[mask].tolist())
                    all_labels[dim].extend(labels[mask].tolist())

        f1s = {}
        for dim in HEAD_SIZES:
            if all_labels[dim]:
                f1s[dim] = f1_score(
                    all_labels[dim],
                    all_preds[dim],
                    average="macro",
                    zero_division=0,  # type: ignore[arg-type]
                )
            else:
                f1s[dim] = 0.0

        avg_f1 = sum(f1s.values()) / len(f1s)
        f1_str = " | ".join(f"{dim}={f1s[dim]:.3f}" for dim in HEAD_SIZES)

        print(
            f"Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f} "
            f"F1=[{f1_str}] avg={avg_f1:.3f} ({elapsed:.0f}s)"
        )

        # Save best
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            save_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "f1": avg_f1,
                    "f1_per_dim": f1s,
                },
                save_path,
            )
            print(f"  Saved best model (F1={avg_f1:.3f})")

    print(f"\nTraining complete. Best avg F1: {best_f1:.3f}")

    # ── ONNX export ──────────────────────────────────────────────────────────

    if args.export_onnx:
        print("\nExporting ONNX...")
        from onnxruntime.quantization import quantize_dynamic, QuantType  # type: ignore[import-not-found]

        model.eval()
        model.cpu()

        dummy_ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long)
        dummy_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long)

        onnx_path = os.path.join(args.out_dir, "deliberation.onnx")

        class ExportWrapper(nn.Module):
            """Wrapper that returns flat tuple for ONNX export."""

            def __init__(self, scorer):
                super().__init__()
                self.scorer = scorer

            def forward(self, input_ids, attention_mask):
                out = self.scorer(input_ids, attention_mask)
                return out["justification"], out["respect"], out["constructiveness"]

        wrapper = ExportWrapper(model)
        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_mask),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["justification", "respect", "constructiveness"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
            },
            opset_version=17,
        )

        # INT8 quantization
        quant_path = os.path.join(args.out_dir, "deliberation_int8.onnx")
        quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QInt8)

        onnx_size = os.path.getsize(onnx_path) / 1e6
        quant_size = os.path.getsize(quant_path) / 1e6
        print(f"ONNX: {onnx_size:.1f} MB -> INT8: {quant_size:.1f} MB")
        print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
