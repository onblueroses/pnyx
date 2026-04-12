"""
Train deliberation quality model on Modal with T4 GPU.

Usage:
    modal run scripts/train_modal.py
"""

import modal

app = modal.App("deliberation-training")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers",
    "sentencepiece",
    "protobuf",
    "scikit-learn",
    "onnx",
    "onnxruntime",
)

vol = modal.Volume.from_name("deliberation-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
    volumes={"/data": vol},
)
def train(epochs: int = 6, batch_size: int = 32, lr: float = 2e-5):
    import json
    import math
    import os
    import random
    import time

    import torch  # type: ignore[import-not-found]
    import torch.nn as nn  # type: ignore[import-not-found]
    from torch.utils.data import DataLoader, Dataset  # type: ignore[import-not-found]

    # No AMP - DeBERTa-v3-small is small enough for FP32 on T4
    from transformers import AutoTokenizer, AutoModel  # type: ignore[import-not-found]
    from sklearn.metrics import f1_score

    MODEL_NAME = "microsoft/deberta-v3-small"
    HIDDEN_SIZE = 768
    MAX_SEQ_LEN = 256
    HEAD_SIZES = {"justification": 4, "respect": 4, "constructiveness": 3}
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    VAL_SPLIT = 0.1
    LOG_EVERY = 100
    OUT_DIR = "/data/output"
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    with open("/data/final.jsonl") as f:
        all_data = [json.loads(line) for line in f]

    print(f"Total samples: {len(all_data)}")
    for dim, n_classes in HEAD_SIZES.items():
        labeled = [r for r in all_data if r.get(dim, -1) >= 0]
        dist = {}
        for r in labeled:
            v = r[dim]
            dist[v] = dist.get(v, 0) + 1
        print(f"  {dim}: {len(labeled)} labeled, dist={dict(sorted(dist.items()))}")

    random.seed(42)
    random.shuffle(all_data)
    val_size = int(len(all_data) * VAL_SPLIT)
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]
    print(f"Train: {len(train_data)}  Val: {len(val_data)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    class DQIDataset(Dataset):
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
        batch_size=batch_size,
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

    class DeliberationScorer(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.norm = nn.LayerNorm(HIDDEN_SIZE)
            self.dropout = nn.Dropout(0.1)
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
            cls = self.dropout(self.norm(out.last_hidden_state[:, 0, :]))
            return {dim: head(cls) for dim, head in self.heads.items()}

    backbone = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    for param in backbone.embeddings.parameters():
        param.requires_grad = False

    model = DeliberationScorer(backbone).float().to(device)
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
    criteria = {dim: nn.CrossEntropyLoss(ignore_index=-1) for dim in HEAD_SIZES}

    print(f"\nTraining: {epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"Schedule: {warmup_steps} warmup / {total_steps} total steps\n")

    best_f1 = 0.0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            loss = sum(
                criteria[dim](logits[dim], batch[dim].to(device)) for dim in HEAD_SIZES
            ) / len(HEAD_SIZES)

            loss.backward()  # type: ignore[union-attr]
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss += loss.item()  # type: ignore[union-attr]
            global_step += 1

            if global_step % LOG_EVERY == 0:
                lr_val = scheduler.get_last_lr()[0]
                print(f"  step {global_step}: loss={loss.item():.4f} lr={lr_val:.2e}")  # type: ignore[union-attr]

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        # Validation
        model.eval()
        all_preds = {dim: [] for dim in HEAD_SIZES}
        all_labels = {dim: [] for dim in HEAD_SIZES}

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
                for dim in HEAD_SIZES:
                    labels = batch[dim]
                    preds = logits[dim].argmax(dim=-1).cpu()
                    mask = labels >= 0
                    all_preds[dim].extend(preds[mask].tolist())
                    all_labels[dim].extend(labels[mask].tolist())

        f1s = {}
        for dim in HEAD_SIZES:
            f1s[dim] = (
                f1_score(
                    all_labels[dim],
                    all_preds[dim],
                    average="macro",
                    zero_division=0,  # type: ignore[arg-type]
                )
                if all_labels[dim]
                else 0.0
            )

        avg_f1 = sum(f1s.values()) / len(f1s)
        f1_str = " | ".join(f"{dim}={f1s[dim]:.3f}" for dim in HEAD_SIZES)
        print(
            f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f} F1=[{f1_str}] avg={avg_f1:.3f} ({elapsed:.0f}s)"
        )

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "model": MODEL_NAME,
                        "hidden_size": HIDDEN_SIZE,
                        "max_seq_len": MAX_SEQ_LEN,
                        "heads": HEAD_SIZES,
                    },
                    "epoch": epoch,
                    "f1": avg_f1,
                    "f1_per_dim": f1s,
                },
                os.path.join(OUT_DIR, "best_model.pt"),
            )
            print(f"  Saved best model (F1={avg_f1:.3f})")

    print(f"\nTraining complete. Best avg F1: {best_f1:.3f}")

    # ONNX Export - fresh model on CPU to avoid FP16 state issues
    from onnxruntime.quantization import quantize_dynamic, QuantType  # type: ignore[import-not-found]

    backbone2 = AutoModel.from_pretrained(MODEL_NAME)
    export_model = DeliberationScorer(backbone2)
    ckpt = torch.load(os.path.join(OUT_DIR, "best_model.pt"), map_location="cpu")
    export_model.load_state_dict(ckpt["model_state_dict"])
    export_model.eval()

    class ExportWrapper(nn.Module):
        def __init__(self, scorer):
            super().__init__()
            self.scorer = scorer

        def forward(self, input_ids, attention_mask):
            out = self.scorer(input_ids, attention_mask)
            return out["justification"], out["respect"], out["constructiveness"]

    wrapper = ExportWrapper(export_model)
    dummy_ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long)
    dummy_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long)

    onnx_path = os.path.join(OUT_DIR, "deliberation.onnx")
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

    quant_path = os.path.join(OUT_DIR, "deliberation_int8.onnx")
    quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QInt8)

    onnx_size = os.path.getsize(onnx_path) / 1e6
    quant_size = os.path.getsize(quant_path) / 1e6
    print(f"ONNX: {onnx_size:.1f} MB -> INT8: {quant_size:.1f} MB")

    # List output artifacts
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f)) / 1e6
        print(f"  {f}: {size:.1f} MB")

    vol.commit()
    return {"best_f1": best_f1, "f1_per_dim": ckpt["f1_per_dim"]}


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/data": vol},
)
def export_onnx():
    import os
    import torch  # type: ignore[import-not-found]
    import torch.nn as nn  # type: ignore[import-not-found]
    from transformers import AutoModel  # type: ignore[import-not-found]
    from onnxruntime.quantization import quantize_dynamic, QuantType  # type: ignore[import-not-found]

    MODEL_NAME = "microsoft/deberta-v3-small"
    HIDDEN_SIZE = 768
    MAX_SEQ_LEN = 256
    HEAD_SIZES = {"justification": 4, "respect": 4, "constructiveness": 3}
    OUT_DIR = "/data/output"

    class DeliberationScorer(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.norm = nn.LayerNorm(HIDDEN_SIZE)
            self.dropout = nn.Dropout(0.1)
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
            cls = self.dropout(self.norm(out.last_hidden_state[:, 0, :]))
            return {dim: head(cls) for dim, head in self.heads.items()}

    backbone = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model = DeliberationScorer(backbone).float()
    ckpt = torch.load(os.path.join(OUT_DIR, "best_model.pt"), map_location="cpu")
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
            return out["justification"], out["respect"], out["constructiveness"]

    wrapper = ExportWrapper(model)
    dummy_ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long)
    dummy_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long)

    onnx_path = os.path.join(OUT_DIR, "deliberation.onnx")
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

    quant_path = os.path.join(OUT_DIR, "deliberation_int8.onnx")
    quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QInt8)

    onnx_size = os.path.getsize(onnx_path) / 1e6
    quant_size = os.path.getsize(quant_path) / 1e6
    print(f"ONNX: {onnx_size:.1f} MB -> INT8: {quant_size:.1f} MB")

    vol.commit()
    return {"onnx_mb": onnx_size, "int8_mb": quant_size}


@app.local_entrypoint()
def main(export_only: bool = False):
    if export_only:
        print("Exporting ONNX from saved checkpoint...")
        result = export_onnx.remote()
    else:
        print("Starting training on T4...")
        result = train.remote()
    print(f"\nResults: {result}")
    print("\nDownload artifacts with:")
    print("  modal volume get deliberation-data output/ ./modal-output/")
