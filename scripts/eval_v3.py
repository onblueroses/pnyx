"""Evaluate Habermas v3 model: F1, calibration, failure modes, temperature sweep.

Usage:
    python3 scripts/eval_v3.py --checkpoint model-output/output-v3/best_model.pt
    python3 scripts/eval_v3.py --checkpoint model-output/output-v3/best_model.pt --temperature 1.5
"""

import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
HIDDEN_SIZE = 768
MAX_SEQ_LEN = 256
HEADS = ["claim_risk", "argument_quality"]


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


def load_model(checkpoint_path):
    backbone = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model = DiscourseScorer(backbone)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt['epoch'] + 1}, F1={ckpt['f1']:.3f}")
    return model


def load_eval_data(data_path, n=1000):
    with open(data_path) as f:
        all_data = [json.loads(line) for line in f]
    random.seed(42)
    random.shuffle(all_data)
    return all_data[:n]


def get_predictions(model, tokenizer, data, temperature=1.0):
    results = {dim: {"probs": [], "preds": [], "labels": []} for dim in HEADS}

    with torch.no_grad():
        for item in data:
            enc = tokenizer(
                item["text"],
                max_length=MAX_SEQ_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            logits = model(enc["input_ids"], enc["attention_mask"])
            for dim in HEADS:
                scaled_logits = logits[dim] / temperature
                prob = torch.softmax(scaled_logits, dim=-1)[0, 1].item()
                pred = 1 if prob >= 0.5 else 0
                results[dim]["probs"].append(prob)
                results[dim]["preds"].append(pred)
                results[dim]["labels"].append(item[dim])

    return results


def print_metrics(results):
    print("\n" + "=" * 60)
    print("CLASSIFICATION METRICS")
    print("=" * 60)

    for dim in HEADS:
        labels = results[dim]["labels"]
        preds = results[dim]["preds"]
        f1 = f1_score(labels, preds, average="macro", zero_division="warn")
        acc = sum(1 for lb, p in zip(labels, preds) if lb == p) / len(labels)
        cm = confusion_matrix(labels, preds)

        print(f"\n{dim}:")
        print(f"  F1 (macro): {f1:.3f}")
        print(f"  Accuracy:   {acc:.1%}")
        print("  Confusion matrix:")
        print("    Predicted ->  0     1")
        print(f"    Actual 0:  {cm[0][0]:5d} {cm[0][1]:5d}")
        print(f"    Actual 1:  {cm[1][0]:5d} {cm[1][1]:5d}")


def print_calibration(results):
    print("\n" + "=" * 60)
    print("CALIBRATION ANALYSIS")
    print("=" * 60)

    for dim in HEADS:
        probs = results[dim]["probs"]
        labels = results[dim]["labels"]

        print(f"\n{dim}:")
        print(
            f"  {'Bucket':>10} | {'Count':>5} | {'Predicted':>9} | {'Actual':>6} | {'Gap':>5}"
        )
        print(f"  {'-' * 10}-+-{'-' * 5}-+-{'-' * 9}-+-{'-' * 6}-+-{'-' * 5}")

        max_gap = 0
        for i in range(10):
            lo, hi = i / 10, (i + 1) / 10
            mask = [(lo <= p < hi) for p in probs]
            count = sum(mask)
            if count == 0:
                continue
            pred_mean = sum(p for p, m in zip(probs, mask) if m) / count
            actual_mean = sum(lb for lb, m in zip(labels, mask) if m) / count
            gap = abs(pred_mean - actual_mean) * 100
            max_gap = max(max_gap, gap)
            print(
                f"  {lo:.1f}-{hi:.1f} | {count:5d} | {pred_mean:8.1%} | {actual_mean:5.1%} | {gap:4.0f}pt"
            )

        print(f"  Max calibration gap: {max_gap:.0f}pt")

        # Uncertainty band
        uncertain = sum(1 for p in probs if 0.1 <= p <= 0.9)
        pct = uncertain / len(probs) * 100
        print(f"  Uncertainty band (0.1-0.9): {uncertain}/{len(probs)} ({pct:.1f}%)")

        # Probability percentiles
        parr = np.array(probs)
        pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        vals = np.percentile(parr, pcts)
        print(
            "  Percentiles: " + ", ".join(f"p{p}={v:.3f}" for p, v in zip(pcts, vals))
        )


def print_failure_modes(results, data):
    print("\n" + "=" * 60)
    print("FAILURE MODE ANALYSIS")
    print("=" * 60)

    # CR false negatives: model says low risk but label is high risk
    cr_probs = results["claim_risk"]["probs"]
    cr_labels = results["claim_risk"]["labels"]
    cr_fn = [
        (i, cr_probs[i])
        for i in range(len(cr_labels))
        if cr_labels[i] == 1 and cr_probs[i] < 0.5
    ]
    cr_fn.sort(key=lambda x: x[1])

    print(f"\nCR False Negatives (misses risky claims): {len(cr_fn)}")
    for idx, prob in cr_fn[:10]:
        print(f"  prob={prob:.3f} | {data[idx]['text'][:100]}")

    # CR false positives
    cr_fp = [
        (i, cr_probs[i])
        for i in range(len(cr_labels))
        if cr_labels[i] == 0 and cr_probs[i] >= 0.5
    ]
    cr_fp.sort(key=lambda x: -x[1])

    print(f"\nCR False Positives (false alarms): {len(cr_fp)}")
    for idx, prob in cr_fp[:10]:
        print(f"  prob={prob:.3f} | {data[idx]['text'][:100]}")

    # AQ false positives: model says has reasoning but label is no reasoning
    aq_probs = results["argument_quality"]["probs"]
    aq_labels = results["argument_quality"]["labels"]
    aq_fp = [
        (i, aq_probs[i])
        for i in range(len(aq_labels))
        if aq_labels[i] == 0 and aq_probs[i] >= 0.5
    ]
    aq_fp.sort(key=lambda x: -x[1])

    print(f"\nAQ False Positives (sees reasoning where there is none): {len(aq_fp)}")
    for idx, prob in aq_fp[:10]:
        print(f"  prob={prob:.3f} | {data[idx]['text'][:100]}")

    # AQ false negatives
    aq_fn = [
        (i, aq_probs[i])
        for i in range(len(aq_labels))
        if aq_labels[i] == 1 and aq_probs[i] < 0.5
    ]
    aq_fn.sort(key=lambda x: x[1])

    print(f"\nAQ False Negatives (misses compressed reasoning): {len(aq_fn)}")
    for idx, prob in aq_fn[:10]:
        print(f"  prob={prob:.3f} | {data[idx]['text'][:100]}")


def temperature_sweep(model, tokenizer, data):
    print("\n" + "=" * 60)
    print("TEMPERATURE SWEEP")
    print("=" * 60)

    temps = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    print(
        f"\n  {'T':>5} | {'CR F1':>5} | {'AQ F1':>5} | {'CR Unc%':>7} | {'AQ Unc%':>7} | {'CR MaxGap':>9} | {'AQ MaxGap':>9}"
    )
    print(
        f"  {'-' * 5}-+-{'-' * 5}-+-{'-' * 5}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 9}-+-{'-' * 9}"
    )

    best_score = -1
    best_t = 1.0

    for t in temps:
        results = get_predictions(model, tokenizer, data, temperature=t)

        metrics = {}
        for dim in HEADS:
            f1 = f1_score(
                results[dim]["labels"],
                results[dim]["preds"],
                average="macro",
                zero_division="warn",
            )
            probs = results[dim]["probs"]
            uncertain = sum(1 for p in probs if 0.1 <= p <= 0.9) / len(probs) * 100

            max_gap = 0
            for i in range(10):
                lo, hi = i / 10, (i + 1) / 10
                mask = [(lo <= p < hi) for p in probs]
                count = sum(mask)
                if count == 0:
                    continue
                pred_mean = sum(p for p, m in zip(probs, mask) if m) / count
                actual_mean = (
                    sum(lb for lb, m in zip(results[dim]["labels"], mask) if m) / count
                )
                gap = abs(pred_mean - actual_mean) * 100
                max_gap = max(max_gap, gap)

            metrics[dim] = {"f1": f1, "uncertain": uncertain, "max_gap": max_gap}

        cr, aq = metrics["claim_risk"], metrics["argument_quality"]
        print(
            f"  {t:5.2f} | {cr['f1']:.3f} | {aq['f1']:.3f} | {cr['uncertain']:6.1f}% | {aq['uncertain']:6.1f}% | {cr['max_gap']:8.0f}pt | {aq['max_gap']:8.0f}pt"
        )

        # Score: F1 must stay above 0.90, then optimize for calibration
        avg_f1 = (cr["f1"] + aq["f1"]) / 2
        avg_gap = (cr["max_gap"] + aq["max_gap"]) / 2
        avg_unc = (cr["uncertain"] + aq["uncertain"]) / 2
        if avg_f1 >= 0.90:
            score = avg_unc - avg_gap  # maximize uncertainty spread, minimize gap
            if score > best_score:
                best_score = score
                best_t = t

    print(f"\n  Recommended temperature: {best_t}")
    return best_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data",
        default="data/training/discourse_quality_10k.jsonl",
        help="Training data (used for temperature sweep)",
    )
    parser.add_argument(
        "--eval-data",
        default=None,
        help="Held-out evaluation data (used for metrics/calibration)",
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sweep", action="store_true", help="Run temperature sweep")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Use held-out eval data if provided, otherwise warn and fall back to training data
    eval_path = args.eval_data or args.data
    if not args.eval_data:
        print(
            "WARNING: No --eval-data provided. Evaluating on training data - metrics may be optimistically biased."
        )
    eval_data = load_eval_data(eval_path, args.n)

    print(
        f"Evaluating on {len(eval_data)} samples from {eval_path} (T={args.temperature})"
    )

    results = get_predictions(model, tokenizer, eval_data, temperature=args.temperature)
    print_metrics(results)
    print_calibration(results)
    print_failure_modes(results, eval_data)

    if args.sweep:
        sweep_data = load_eval_data(args.data, args.n)
        print(f"\nTemperature sweep on training data ({len(sweep_data)} samples)...")
        best_t = temperature_sweep(model, tokenizer, sweep_data)
        print(f"\nRe-running eval with recommended T={best_t}...")
        results = get_predictions(model, tokenizer, eval_data, temperature=best_t)
        print_metrics(results)
        print_calibration(results)


if __name__ == "__main__":
    main()
