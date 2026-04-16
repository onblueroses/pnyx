"""
Train discourse quality model v3 on Modal with T4 GPU.

v3 changes from v2:
  - Focal loss (gamma=2) + label smoothing (0.05) to fix overconfidence
  - Resumes from v2 checkpoint by default
  - All layers unfrozen, lr=5e-6
  - Per-epoch calibration logging (10-bin table + uncertainty band %)
  - FP16 ONNX export (single file, no INT8 attempt)

Usage:
    # Upload merged dataset
    modal volume put deliberation-data data/training/discourse_quality_v3.jsonl discourse_quality_v3.jsonl

    # Train (resumes from v2 by default)
    modal run scripts/train_v3_modal.py

    # Download results
    modal volume get deliberation-data output-v3/ ./model-output/output-v3/
"""

import modal

app = modal.App("discourse-training-v3")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers",
    "sentencepiece",
    "protobuf",
    "scikit-learn",
    "onnx",
    "onnxruntime",
    "onnxscript",
    "numpy",
)

vol = modal.Volume.from_name("deliberation-data", create_if_missing=True)

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
HIDDEN_SIZE = 768
MAX_SEQ_LEN = 256
HEADS = {"claim_risk": 2, "argument_quality": 2}


@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
    volumes={"/data": vol},
)
def train(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 5e-6,
    unfreeze_all: bool = True,
    resume_from: str = "output-v2/best_model.pt",
    data_file: str = "discourse_quality_v3.jsonl",
):
    import json
    import math
    import os
    import time

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics import f1_score, confusion_matrix

    OUT_DIR = "/data/output-v3"
    ONNX_DIR = os.path.join(OUT_DIR, "onnx")
    os.makedirs(ONNX_DIR, exist_ok=True)

    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    config = {
        "model": MODEL_NAME,
        "hidden_size": HIDDEN_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "heads": HEADS,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "unfreeze_all": unfreeze_all,
    }
    print(f"Config: {json.dumps(config, indent=2)}")

    # Load data
    with open(f"/data/{data_file}") as f:
        all_data = [json.loads(line) for line in f]

    print(f"Total samples: {len(all_data)}")
    for dim in HEADS:
        dist = {}
        for r in all_data:
            v = r[dim]
            dist[v] = dist.get(v, 0) + 1
        print(f"  {dim}: dist={dict(sorted(dist.items()))}")

    import hashlib

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
        batch_size=batch_size,
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

    if not unfreeze_all:
        for param in backbone.embeddings.parameters():
            param.requires_grad = False
        print("Frozen: embeddings only")
    else:
        print("All layers unfrozen")

    model = DiscourseScorer(backbone).float()

    if resume_from:
        ckpt_path = os.path.join("/data", resume_from)
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded epoch {ckpt['epoch'] + 1}, F1={ckpt['f1']:.3f}")

    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {trainable:,} trainable / {total_params:,} total")

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, label_smoothing=0.05):
            super().__init__()
            self.gamma = gamma
            self.ce = nn.CrossEntropyLoss(
                reduction="none", label_smoothing=label_smoothing
            )

        def forward(self, logits, targets):
            ce_loss = self.ce(logits, targets)
            pt = torch.exp(-ce_loss)
            return ((1 - pt) ** self.gamma * ce_loss).mean()

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)

    print(f"\nTraining: {epochs} epochs, {len(train_loader)} batches/epoch")
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

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            loss = (
                criterion(logits["claim_risk"], batch["claim_risk"].to(device))
                + criterion(
                    logits["argument_quality"], batch["argument_quality"].to(device)
                )
            ) / 2

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                lr_val = scheduler.get_last_lr()[0]
                print(f"  step {global_step}: loss={loss.item():.4f} lr={lr_val:.2e}")

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        # Validation
        model.eval()
        all_preds = {dim: [] for dim in HEADS}
        all_labels = {dim: [] for dim in HEADS}
        all_probs = {dim: [] for dim in HEADS}

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
                for dim in HEADS:
                    probs = torch.softmax(logits[dim], dim=-1)[:, 1].cpu()
                    preds = logits[dim].argmax(dim=-1).cpu()
                    all_preds[dim].extend(preds.tolist())
                    all_labels[dim].extend(batch[dim].tolist())
                    all_probs[dim].extend(probs.tolist())

        f1s = {}
        cms = {}
        for dim in HEADS:
            f1s[dim] = f1_score(
                all_labels[dim], all_preds[dim], average="macro", zero_division="warn"
            )
            cms[dim] = confusion_matrix(all_labels[dim], all_preds[dim])

        avg_f1 = sum(f1s.values()) / len(f1s)
        f1_str = " | ".join(f"{dim}={f1s[dim]:.3f}" for dim in HEADS)

        # Calibration analysis
        for dim in HEADS:
            probs = all_probs[dim]
            labels = all_labels[dim]
            bins = [(i / 10, (i + 1) / 10) for i in range(10)]
            print(f"  Calibration [{dim}]:")
            print(
                f"    {'Bucket':>10} | {'Count':>5} | {'Predicted':>9} | {'Actual':>6} | {'Gap':>5}"
            )
            for lo, hi in bins:
                mask = [(lo <= p < hi) for p in probs]
                count = sum(mask)
                if count == 0:
                    continue
                pred_mean = sum(p for p, m in zip(probs, mask) if m) / count
                actual_mean = sum(lb for lb, m in zip(labels, mask) if m) / count
                gap = abs(pred_mean - actual_mean) * 100
                print(
                    f"    {lo:.1f}-{hi:.1f} | {count:5d} | {pred_mean:8.1%} | {actual_mean:5.1%} | {gap:4.0f}pt"
                )

        # Uncertainty band: % of predictions in 0.1-0.9 range
        for dim in HEADS:
            probs = all_probs[dim]
            uncertain = sum(1 for p in probs if 0.1 <= p <= 0.9)
            pct = uncertain / len(probs) * 100
            print(f"  Uncertainty band [{dim}]: {uncertain}/{len(probs)} ({pct:.1f}%)")

        print(
            f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f} F1=[{f1_str}] avg={avg_f1:.3f} ({elapsed:.0f}s)"
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
                os.path.join(OUT_DIR, "best_model.pt"),
            )
            print(f"  Saved best model (F1={avg_f1:.3f})")

    # Confusion matrices
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

    # ONNX Export (FP16, single file - no INT8, see D18)
    print("\nExporting ONNX (FP16)...")
    import onnx
    import onnxruntime as ort

    backbone2 = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    export_model = DiscourseScorer(backbone2)
    ckpt = torch.load(
        os.path.join(OUT_DIR, "best_model.pt"), map_location="cpu", weights_only=True
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

    model_fp16 = export_model.half()
    wrapper = ExportWrapper(model_fp16)
    dummy_ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long)
    dummy_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long)

    onnx_path_tmp = os.path.join(ONNX_DIR, "model_tmp.onnx")
    onnx_path = os.path.join(ONNX_DIR, "model.onnx")
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        onnx_path_tmp,
        input_names=["input_ids", "attention_mask"],
        output_names=["claim_risk", "argument_quality"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )

    # Consolidate external data into single file
    onnx_model = onnx.load(onnx_path_tmp, load_external_data=True)
    onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)
    for f in os.listdir(ONNX_DIR):
        if f.startswith("model_tmp"):
            os.remove(os.path.join(ONNX_DIR, f))

    onnx_size = os.path.getsize(onnx_path) / 1e6
    print(f"ONNX (FP16, single file): {onnx_size:.1f} MB")

    log_lines.append("## ONNX Export\n")
    log_lines.append(f"- FP16 model: {onnx_size:.1f} MB\n")

    # Save tokenizer for Transformers.js
    tokenizer.save_pretrained(ONNX_DIR)
    print(f"Tokenizer saved to {ONNX_DIR}")

    # Config for Transformers.js
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
    with open(os.path.join(ONNX_DIR, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    # Verify ONNX matches PyTorch
    print("\nVerifying ONNX output...")
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
        log_lines.append(f"- {name} ONNX verification: max_diff={diff:.6f} [{status}]")

    # Write training log
    log_path = os.path.join(OUT_DIR, "training_log.md")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    # List artifacts
    print("\nArtifacts:")
    for root, dirs, files in os.walk(OUT_DIR):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            size = os.path.getsize(fpath) / 1e6
            rel = os.path.relpath(fpath, OUT_DIR)
            print(f"  {rel}: {size:.1f} MB")

    vol.commit()
    return {"best_f1": best_f1, "f1_per_dim": ckpt["f1_per_dim"]}


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/data": vol},
)
def export_onnx():
    import json
    import os

    import numpy as np
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    import onnxruntime as ort

    OUT_DIR = "/data/output-v3"
    ONNX_DIR = os.path.join(OUT_DIR, "onnx")
    os.makedirs(ONNX_DIR, exist_ok=True)

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
    model = DiscourseScorer(backbone)
    ckpt = torch.load(
        os.path.join(OUT_DIR, "best_model.pt"), map_location="cpu", weights_only=True
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: epoch {ckpt['epoch'] + 1}, F1={ckpt['f1']:.3f}")
    print(f"Per-dim: {ckpt['f1_per_dim']}")

    class ExportWrapper(nn.Module):
        def __init__(self, scorer):
            super().__init__()
            self.scorer = scorer

        def forward(self, input_ids, attention_mask):
            out = self.scorer(input_ids, attention_mask)
            return out["claim_risk"], out["argument_quality"]

    # Convert to FP16 for smaller ONNX file
    model_fp16 = model.half()
    wrapper = ExportWrapper(model_fp16)
    dummy_ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long)
    dummy_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long)

    onnx_path_tmp = os.path.join(ONNX_DIR, "model_tmp.onnx")
    onnx_path = os.path.join(ONNX_DIR, "model.onnx")
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        onnx_path_tmp,
        input_names=["input_ids", "attention_mask"],
        output_names=["claim_risk", "argument_quality"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )

    # Consolidate external data into single file
    import onnx

    onnx_model = onnx.load(onnx_path_tmp, load_external_data=True)
    onnx.save_model(
        onnx_model,
        onnx_path,
        save_as_external_data=False,
    )
    # Clean up tmp files
    for f in os.listdir(ONNX_DIR):
        if f.startswith("model_tmp"):
            os.remove(os.path.join(ONNX_DIR, f))

    onnx_size = os.path.getsize(onnx_path) / 1e6
    print(f"ONNX (FP16, single file): {onnx_size:.1f} MB")

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(ONNX_DIR)

    # Config for Transformers.js
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
    with open(os.path.join(ONNX_DIR, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    # Verify ONNX matches PyTorch
    print("\nVerifying ONNX output...")
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

    # List artifacts
    print("\nArtifacts:")
    for root, dirs, files in os.walk(OUT_DIR):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            size = os.path.getsize(fpath) / 1e6
            rel = os.path.relpath(fpath, OUT_DIR)
            print(f"  {rel}: {size:.1f} MB")

    vol.commit()
    return {"onnx_mb": onnx_size}


@app.local_entrypoint()
def main(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 5e-6,
    unfreeze_all: bool = True,
    export_only: bool = False,
    resume_from: str = "output-v2/best_model.pt",
    data_file: str = "discourse_quality_v3.jsonl",
):
    if export_only:
        print("Exporting ONNX from saved checkpoint...")
        result = export_onnx.remote()
    else:
        print(
            f"Starting training on Modal (epochs={epochs}, lr={lr}, unfreeze_all={unfreeze_all}, resume={resume_from}, data={data_file})..."
        )
        result = train.remote(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            unfreeze_all=unfreeze_all,
            resume_from=resume_from,
            data_file=data_file,
        )
    print(f"\nResults: {result}")
    print("\nDownload artifacts with:")
    print("  modal volume get deliberation-data output-v3/ ./model-output/output-v3/")
