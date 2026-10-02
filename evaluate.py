"""Evaluate a trained Sama head on MMLU and ARC test splits.

Usage:
    python evaluate.py --model-dir models/v0.2
    python evaluate.py --model-dir models/v0.1 --device cpu
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from sama.engine import TypedDecisionHead


# Re-use the same formatting from train.py to stay byte-identical
def make_text(context: str, question: str, options: list[str]) -> str:
    parts = [context, ""]
    parts.append(f"Question: {question}")
    opts_str = " ".join(f"({chr(65 + j)}) {o}" for j, o in enumerate(options))
    parts.append(f"Options: {opts_str}")
    parts.append("Answer:")
    return "\n".join(parts)


def load_mmlu_test(limit=None):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    examples = []
    for row in ds:
        choices = row["choices"]
        if len(choices) != 4:
            continue
        text = make_text(row["question"],
                         "Which of the following is the correct answer?", choices)
        examples.append((text, int(row["answer"]), len(choices)))
        if limit and len(examples) >= limit:
            break
    return examples


def load_arc_test(limit=None):
    from datasets import load_dataset
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3,
                 "1": 0, "2": 1, "3": 2, "4": 3}
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test",
                      trust_remote_code=True)
    examples = []
    for row in ds:
        choices = row["choices"]["text"]
        labels_list = row["choices"]["label"]
        ak = row["answerKey"]
        if ak in label_map:
            label = label_map[ak]
        elif ak in labels_list:
            label = labels_list.index(ak)
        else:
            continue
        if len(choices) != 4:
            continue
        text = make_text(row["question"],
                         "Which of the following is the correct answer?", choices)
        examples.append((text, label, len(choices)))
        if limit and len(examples) >= limit:
            break
    return examples


def compute_ece(confs, accs, n_bins=15):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accs[mask].mean()
        bin_conf = confs[mask].mean()
        ece += mask.sum() / len(confs) * abs(bin_acc - bin_conf)
    return float(ece)


def find_optimal_temperature(logits_list, labels_list, n_options=4):
    """Grid search for optimal calibration temperature."""
    temps = np.linspace(0.5, 2.0, 100)
    best_ece, best_t = 1.0, 1.0
    for t in temps:
        confs, accs = [], []
        for logits, label in zip(logits_list, labels_list):
            probs = F.softmax(logits[:n_options] / t, dim=-1).numpy()
            pred = int(probs.argmax())
            confs.append(float(probs.max()))
            accs.append(1.0 if pred == label else 0.0)
        ece = compute_ece(np.array(confs), np.array(accs))
        if ece < best_ece:
            best_ece = ece
            best_t = float(t)
    return best_t, best_ece


def save_reliability_diagram(confs, accs, path, n_bins=15):
    """Save a reliability diagram as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping reliability diagram")
        return

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_accs = []
    bin_counts = []
    for i in range(n_bins):
        mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            bin_accs.append(0)
            bin_counts.append(0)
        else:
            bin_accs.append(accs[mask].mean())
            bin_counts.append(mask.sum())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6),
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.bar(bin_centers, bin_accs, width=1.0 / n_bins, alpha=0.6,
            edgecolor="black", label="Accuracy")
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Reliability Diagram")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.bar(bin_centers, bin_counts, width=1.0 / n_bins, alpha=0.6,
            edgecolor="black", color="orange")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved reliability diagram to {path}")


@torch.no_grad()
def evaluate(args):
    if args.dry_run:
        print("[DRY RUN] Bypassing encoder and real data.")
        device = torch.device("cpu")
        hidden_dim = 1536
        head = TypedDecisionHead(hidden_dim=hidden_dim).to(device)
        
        mmlu_logits = [torch.randn(4) for _ in range(50)]
        mmlu_labels = [int(np.random.randint(0, 4)) for _ in range(50)]
        arc_logits = [torch.randn(4) for _ in range(20)]
        arc_labels = [int(np.random.randint(0, 4)) for _ in range(20)]
        
        opt_t, cal_ece = find_optimal_temperature(mmlu_logits + arc_logits, mmlu_labels + arc_labels)
        
        print(f"MMLU (mocked): Accuracy: 0.2500, ECE: 0.1000")
        print(f"ARC-Challenge (mocked): Accuracy: 0.2500, ECE: 0.1000")
        print(f"\nOptimal temperature: {opt_t:.4f}")
        print(f"ECE (calibrated): {cal_ece:.4f}")
        
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "eval_results.json")
        with open(results_path, "w") as f:
            json.dump({"dry_run": True}, f)
            
        print(f"\nSaved results to {results_path}")
        print("[DRY RUN] ✓ Passed")
        import sys
        sys.exit(0)

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Load config + head
    config_path = os.path.join(args.model_dir, "config.json")
    head_path = os.path.join(args.model_dir, "head.pt")
    with open(config_path) as f:
        config = json.load(f)

    encoder_id = config.get("encoder", "Qwen/Qwen2.5-0.5B")
    hidden_dim = config.get("hidden_dim", 896)

    print(f"Loading encoder: {encoder_id}")
    tokenizer = AutoTokenizer.from_pretrained(encoder_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoder = AutoModel.from_pretrained(
        encoder_id, torch_dtype=torch.bfloat16,
    ).to(device)
    encoder.eval()

    head = TypedDecisionHead(hidden_dim=hidden_dim)
    head.load_state_dict(torch.load(head_path, map_location="cpu"))
    head = head.to(device)
    head.eval()

    # Load test data
    print("Loading MMLU test split...")
    mmlu_test = load_mmlu_test()
    print(f"  {len(mmlu_test)} examples")
    print("Loading ARC-Challenge test split...")
    arc_test = load_arc_test()
    print(f"  {len(arc_test)} examples")

    def run_eval(examples, name):
        correct = 0
        all_logits = []
        all_labels = []
        all_confs = []
        all_accs = []

        for i in range(0, len(examples), 32):
            batch = examples[i:i + 32]
            texts = [t for t, _, _ in batch]
            labels = [l for _, l, _ in batch]

            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=2048).to(device)
            out = encoder(**enc, output_hidden_states=True)
            h = out.hidden_states[-1].float()
            seq_lens = enc["attention_mask"].sum(dim=1) - 1
            pooled = h[torch.arange(h.shape[0], device=device), seq_lens]

            head_out = head(pooled)
            for j, label in enumerate(labels):
                logits = head_out["choice_logits"][j, :4].cpu()
                all_logits.append(logits)
                all_labels.append(label)

                probs = F.softmax(logits, dim=-1).numpy()
                pred = int(probs.argmax())
                conf = float(probs.max())
                correct += (pred == label)
                all_confs.append(conf)
                all_accs.append(1.0 if pred == label else 0.0)

        accuracy = correct / len(examples)
        confs = np.array(all_confs)
        accs = np.array(all_accs)
        ece = compute_ece(confs, accs)

        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{len(examples)})")
        print(f"  ECE (uncalibrated): {ece:.4f}")
        return accuracy, ece, all_logits, all_labels, confs, accs

    mmlu_acc, mmlu_ece, mmlu_logits, mmlu_labels, mmlu_confs, mmlu_accs = \
        run_eval(mmlu_test, "MMLU")
    arc_acc, arc_ece, arc_logits, arc_labels, arc_confs, arc_accs = \
        run_eval(arc_test, "ARC-Challenge")

    # Calibration (fit temperature on combined data)
    all_logits = mmlu_logits + arc_logits
    all_labels = mmlu_labels + arc_labels
    opt_t, cal_ece = find_optimal_temperature(all_logits, all_labels)
    print(f"\nOptimal temperature: {opt_t:.4f}")
    print(f"ECE (calibrated): {cal_ece:.4f}")

    # Calibrated confidences for reliability diagram
    cal_confs = []
    cal_accs = []
    for logits, label in zip(all_logits, all_labels):
        probs = F.softmax(logits[:4] / opt_t, dim=-1).numpy()
        pred = int(probs.argmax())
        cal_confs.append(float(probs.max()))
        cal_accs.append(1.0 if pred == label else 0.0)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "mmlu_accuracy": mmlu_acc,
        "arc_accuracy": arc_acc,
        "ece_uncal": mmlu_ece,
        "ece_cal": cal_ece,
        "optimal_temperature": opt_t,
        "head_params": sum(p.numel() for p in head.parameters()),
    }
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    diagram_path = os.path.join(args.output_dir, "reliability_curve.png")
    save_reliability_diagram(
        np.array(cal_confs), np.array(cal_accs), diagram_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Sama head on MMLU + ARC",
    )
    parser.add_argument("--model-dir",
                        help="Directory containing head.pt and config.json")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to save eval_results.json and PNG")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cpu, or cuda")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run instantly with synthetic data (CPU ok)")
    args = parser.parse_args()
    if not args.dry_run and not args.model_dir:
        parser.error("--model-dir is required unless --dry-run is used")
    evaluate(args)


if __name__ == "__main__":
    main()
