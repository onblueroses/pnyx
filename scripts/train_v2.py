#!/usr/bin/env python3
"""
Discourse Quality Model v2 - DeBERTa-v3-small (NLI) with 2 binary heads.

Fine-tunes cross-encoder/nli-deberta-v3-small for claim_risk and argument_quality
binary classification. Exports to ONNX with tokenizer files for Transformers.js.

Usage:
    python train_v2.py --data data/training/discourse_quality_10k.jsonl
    python train_v2.py --data data/training/discourse_quality_10k.jsonl --unfreeze-all --lr 1e-5
    python train_v2.py --dry-run --data data/training/discourse_quality_10k.jsonl
"""

import argparse
import json
import math
import os


MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
HIDDEN_SIZE = 768
MAX_SEQ_LEN = 256
HEADS = {"claim_risk": 2, "argument_quality": 2}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", required=True, help="Path to discourse_quality_10k.jsonl")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--out-dir", default="model-output")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument(
        "--unfreeze-all",
        action="store_true",
        help="Unfreeze all layers including embeddings",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export")
    return p.parse_args()


def load_data(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    args = parse_args()

    config = {
        "model": MODEL_NAME,
        "hidden_size": HIDDEN_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "heads": HEADS,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "unfreeze_all": args.unfreeze_all,
    }
    print(f"Config: {json.dumps(config, indent=2)}")

    all_data = load_data(args.data)
    print(f"Total samples: {len(all_data)}")

    for dim in HEADS:
        dist = {}
        for r in all_data:
            v = r[dim]
            dist[v] = dist.get(v, 0) + 1
        print(f"  {dim}: dist={dict(sorted(dist.items()))}")

    if args.dry_run:
        print("Dry run - exiting.")
        return

    import random
    import time

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torch.amp.autocast_mode import autocast
    from torch.amp.grad_scaler import GradScaler
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics import f1_score, confusion_matrix

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}")
    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "onnx"), exist_ok=True)

    # Train/val split
    random.seed(42)
    random.shuffle(all_data)
    val_size = int(len(all_data) * args.val_split)
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]
    print(f"Train: {len(train_data)}  Val: {len(val_data)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    class DiscourseDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            enc = tokenizer(
                item["text"],
                max_length=MAX_SEQ_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "claim_risk": torch.tensor(item["claim_risk"], dtype=torch.long),
                "argument_quality": torch.tensor(
                    item["argument_quality"], dtype=torch.long
                ),
            }

    train_loader = DataLoader(
        DiscourseDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        DiscourseDataset(val_data),
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    class DiscourseScorer(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.dropout = nn.Dropout(0.1)
            self.norm = nn.LayerNorm(HIDDEN_SIZE)

            self.claim_risk_head = nn.Sequential(
                nn.Linear(HIDDEN_SIZE, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 2),
            )

            self.argument_quality_head = nn.Sequential(
                nn.Linear(HIDDEN_SIZE, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 2),
            )

        def forward(self, input_ids, attention_mask):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            cls = self.dropout(self.norm(out.last_hidden_state[:, 0, :]))
            return {
                "claim_risk": self.claim_risk_head(cls),
                "argument_quality": self.argument_quality_head(cls),
            }

    backbone = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

    if not args.unfreeze_all:
        for param in backbone.embeddings.parameters():
            param.requires_grad = False
        print("Frozen: embeddings only")
    else:
        print("All layers unfrozen")

    model = DiscourseScorer(backbone).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {trainable:,} trainable / {total_params:,} total")

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
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=use_amp)

    print(f"\nTraining: {args.epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"Schedule: {warmup_steps} warmup / {total_steps} total steps\n")

    log_lines = []
    log_lines.append("# Training Log\n")
    log_lines.append(f"## Config\n```json\n{json.dumps(config, indent=2)}\n```\n")
    log_lines.append("## Per-Epoch Metrics\n")
    log_lines.append(
        "| Epoch | Loss | claim_risk F1 | argument_quality F1 | Avg F1 | Time |"
    )
    log_lines.append(
        "|-------|------|---------------|---------------------|--------|------|"
    )

    best_f1 = 0.0
    best_epoch_data = {}
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast("cuda", enabled=use_amp):
                logits = model(input_ids, attention_mask)
                loss = (
                    criterion(logits["claim_risk"], batch["claim_risk"].to(device))
                    + criterion(
                        logits["argument_quality"], batch["argument_quality"].to(device)
                    )
                ) / 2

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                lr_val = scheduler.get_last_lr()[0]
                print(f"  step {global_step}: loss={loss.item():.4f} lr={lr_val:.2e}")

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        # Validation
        model.eval()
        all_preds = {dim: [] for dim in HEADS}
        all_labels = {dim: [] for dim in HEADS}

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with autocast("cuda", enabled=use_amp):
                    logits = model(input_ids, attention_mask)
                for dim in HEADS:
                    preds = logits[dim].argmax(dim=-1).cpu()
                    all_preds[dim].extend(preds.tolist())
                    all_labels[dim].extend(batch[dim].tolist())

        f1s = {}
        cms = {}
        for dim in HEADS:
            f1s[dim] = f1_score(
                all_labels[dim], all_preds[dim], average="macro", zero_division="warn"
            )
            cms[dim] = confusion_matrix(all_labels[dim], all_preds[dim])

        avg_f1 = sum(f1s.values()) / len(f1s)
        f1_str = " | ".join(f"{dim}={f1s[dim]:.3f}" for dim in HEADS)

        print(
            f"Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f} F1=[{f1_str}] avg={avg_f1:.3f} ({elapsed:.0f}s)"
        )

        log_lines.append(
            f"| {epoch + 1} | {avg_loss:.4f} | {f1s['claim_risk']:.4f} | {f1s['argument_quality']:.4f} | {avg_f1:.4f} | {elapsed:.0f}s |"
        )

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_epoch_data = {
                "epoch": epoch + 1,
                "f1s": dict(f1s),
                "cms": {d: cm.tolist() for d, cm in cms.items()},
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "f1": avg_f1,
                    "f1_per_dim": f1s,
                },
                os.path.join(args.out_dir, "best_model.pt"),
            )
            print(f"  Saved best model (F1={avg_f1:.3f})")

    # Final confusion matrices
    log_lines.append(f"\n## Best Model (Epoch {best_epoch_data.get('epoch', '?')})\n")
    log_lines.append(f"Average F1: {best_f1:.4f}\n")
    for dim in HEADS:
        cm = best_epoch_data.get("cms", {}).get(dim, [])
        log_lines.append(f"### {dim} Confusion Matrix")
        log_lines.append("```")
        log_lines.append("Predicted ->  0    1")
        if cm:
            log_lines.append(f"Actual 0:  {cm[0][0]:4d} {cm[0][1]:4d}")
            log_lines.append(f"Actual 1:  {cm[1][0]:4d} {cm[1][1]:4d}")
        log_lines.append("```\n")

    print(f"\nTraining complete. Best avg F1: {best_f1:.3f}")

    # ONNX export
    if not args.skip_onnx:
        print("\nExporting ONNX...")

        # Reload on CPU for clean export
        backbone2 = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        export_model = DiscourseScorer(backbone2)
        ckpt = torch.load(
            os.path.join(args.out_dir, "best_model.pt"),
            map_location="cpu",
            weights_only=True,
        )
        export_model.load_state_dict(ckpt["model_state_dict"])
        export_model.eval()

        class ExportWrapper(nn.Module):
            def __init__(self, scorer):
                super().__init__()
                self.scorer = scorer

            def forward(self, input_ids, attention_mask):
                out = self.scorer(input_ids, attention_mask)
                return out["claim_risk"], out["argument_quality"]

        wrapper = ExportWrapper(export_model)
        dummy_ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long)
        dummy_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long)

        onnx_dir = os.path.join(args.out_dir, "onnx")
        onnx_path = os.path.join(onnx_dir, "model.onnx")

        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_mask),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["claim_risk", "argument_quality"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
            },
            opset_version=17,
        )

        # INT8 quantization skipped: DeBERTa ShapeInferenceError (768 vs 256 dim mismatch)
        # See DECISIONS.md D18 for details
        onnx_size = os.path.getsize(onnx_path) / 1e6
        print(f"ONNX: {onnx_size:.1f} MB (FP32, INT8 skipped - DeBERTa incompatible)")

        log_lines.append("## ONNX Export\n")
        log_lines.append(f"- Full model: {onnx_size:.1f} MB (FP32)")
        log_lines.append("- INT8: skipped (DeBERTa ShapeInferenceError)\n")

        # Save tokenizer files for Transformers.js
        tokenizer.save_pretrained(onnx_dir)
        print(f"Tokenizer files saved to {onnx_dir}")

        # Save config.json for Transformers.js compatibility
        model_config = {
            "architectures": ["DiscourseScorer"],
            "model_type": "deberta-v2",
            "hidden_size": HIDDEN_SIZE,
            "max_position_embeddings": MAX_SEQ_LEN,
            "num_labels": 2,
            "id2label": {"0": "low", "1": "high"},
            "label2id": {"low": 0, "high": 1},
            "output_names": ["claim_risk", "argument_quality"],
        }
        with open(os.path.join(onnx_dir, "config.json"), "w") as f:
            json.dump(model_config, f, indent=2)

        # Verify ONNX output matches PyTorch
        print("\nVerifying ONNX output...")
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path)
        pt_out = wrapper(dummy_ids, dummy_mask)
        onnx_out = sess.run(
            None,
            {
                "input_ids": dummy_ids.numpy(),
                "attention_mask": dummy_mask.numpy(),
            },
        )

        for i, name in enumerate(["claim_risk", "argument_quality"]):
            diff = np.abs(pt_out[i].detach().numpy() - onnx_out[i]).max()
            status = "PASS" if diff < 1e-4 else "FAIL"
            print(f"  {name}: max diff={diff:.6f} [{status}]")
            log_lines.append(
                f"- {name} ONNX verification: max_diff={diff:.6f} [{status}]"
            )

    # Write training log
    log_path = os.path.join(args.out_dir, "training_log.md")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nTraining log: {log_path}")

    # List artifacts
    print("\nArtifacts:")
    for root, dirs, files in os.walk(args.out_dir):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            size = os.path.getsize(fpath) / 1e6
            rel = os.path.relpath(fpath, args.out_dir)
            print(f"  {rel}: {size:.1f} MB")


if __name__ == "__main__":
    main()
