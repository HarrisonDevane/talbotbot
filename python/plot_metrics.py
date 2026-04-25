#!/usr/bin/env python3
"""
Plot Talbot training metrics and opening diversity from a single rl_dir.

Outputs:
  - <rl_dir>/opening_trends.png   (opening move diversity over training)
  - <rl_dir>/training_metrics.png  (loss, entropy, grad norm, etc.)

Usage:
    python talbot_plots.py <rl_dir>
    python talbot_plots.py /d/Projects/talbot/rl_dir
"""

import re
import sys
import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Opening diversity
# =============================================================================

def parse_openings(folder):
    """Parse all worker_*.log files in a folder for first moves."""
    files = sorted(glob.glob(os.path.join(folder, "worker_*.log")))
    if not files:
        return None, None, 0

    white = defaultdict(int)
    black = defaultdict(int)
    total = 0

    for f in files:
        with open(f) as fh:
            content = fh.read()
        for _, moves in re.findall(
            r'\[Result "([^"]+)"\].*?\n\n(1\. .+?)(?:\n\n|\Z)', content, re.DOTALL
        ):
            m = re.match(r"1\. (\S+) (\S+)", moves.replace("\n", " "))
            if m:
                white[m.group(1)] += 1
                black[m.group(2)] += 1
                total += 1

    return white, black, total


def plot_openings(parent, step_dirs):
    """Generate opening_trends.png."""
    steps, w_main, w_junk, b_princ, b_junk = [], [], [], [], []
    w_d4, w_e4, w_nf3, w_c4 = [], [], [], []
    b_d5, b_e5, b_c5, b_nf6 = [], [], [], []

    for sd in step_dirs:
        try:
            step = int(os.path.basename(sd).replace("run_step_", ""))
        except ValueError:
            continue

        white, black, total = parse_openings(sd)
        if total == 0:
            continue

        steps.append(step)
        w_main.append(sum(white.get(m, 0) for m in ["d4", "e4", "Nf3", "c4"]) / total * 100)
        w_junk.append(sum(white.get(m, 0) for m in ["Na3", "Nh3", "h4", "g4", "a3", "a4", "f3"]) / total * 100)
        b_princ.append(sum(black.get(m, 0) for m in ["d5", "Nf6", "c5", "e5", "e6", "d6", "Nc6", "g6"]) / total * 100)
        b_junk.append(sum(black.get(m, 0) for m in ["a6", "h6", "a5", "h5", "Na6", "Nh6", "g5", "f6"]) / total * 100)
        w_d4.append(white.get("d4", 0) / total * 100)
        w_e4.append(white.get("e4", 0) / total * 100)
        w_nf3.append(white.get("Nf3", 0) / total * 100)
        w_c4.append(white.get("c4", 0) / total * 100)
        b_d5.append(black.get("d5", 0) / total * 100)
        b_e5.append(black.get("e5", 0) / total * 100)
        b_c5.append(black.get("c5", 0) / total * 100)
        b_nf6.append(black.get("Nf6", 0) / total * 100)

        print(f"  Openings step {step:>6}: {total:>5} games | W-main {w_main[-1]:5.1f}% | B-princ {b_princ[-1]:5.1f}%")

    if not steps:
        print("  No opening data found, skipping opening_trends.png")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Talbot Opening Diversity Over Training", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(steps, w_main, "o-", ms=3, label="White mainline (d4+e4+Nf3+c4)")
    ax.plot(steps, b_princ, "s-", ms=3, label="Black principled (d5+Nf6+c5+e5+e6+d6+Nc6+g6)")
    ax.set_ylabel("%")
    ax.set_title("Good Openings %")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, w_junk, "o-", ms=3, color="red", label="White junk")
    ax.plot(steps, b_junk, "s-", ms=3, color="orange", label="Black junk")
    ax.set_ylabel("%")
    ax.set_title("Junk Openings %")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, w_d4, "o-", ms=3, label="d4")
    ax.plot(steps, w_e4, "s-", ms=3, label="e4")
    ax.plot(steps, w_nf3, "^-", ms=3, label="Nf3")
    ax.plot(steps, w_c4, "D-", ms=3, label="c4")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("%")
    ax.set_title("White Individual Openings")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, b_d5, "o-", ms=3, label="d5")
    ax.plot(steps, b_e5, "s-", ms=3, label="e5")
    ax.plot(steps, b_c5, "^-", ms=3, label="c5")
    ax.plot(steps, b_nf6, "D-", ms=3, label="Nf6")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("%")
    ax.set_title("Black Individual Openings")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(parent, "opening_trends.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# =============================================================================
# Training metrics
# =============================================================================

def parse_training_log(filepath):
    """Parse a single training_run.log."""
    rows = []
    with open(filepath) as f:
        for line in f:
            m = re.search(
                r"Step (\d+) \| Loss: T=([\d.]+) \(P=([\d.]+), V=([\d.]+)\) "
                r"\| Batch Vals \(W/D/L\): ([\d.]+)% / ([\d.]+)% / ([\d.]+)%"
                r"\| P_Ent=([\d.]+) "
                r"\| V_Out=([-\d.]+) \| V_Tar=([-\d.]+) "
                r"\| Grad=([\d.]+) \| LR=([\d.]+)",
                line,
            )
            if m:
                rows.append({
                    "step": int(m.group(1)),
                    "total_loss": float(m.group(2)),
                    "policy_loss": float(m.group(3)),
                    "value_loss": float(m.group(4)),
                    "batch_w": float(m.group(5)),
                    "batch_d": float(m.group(6)),
                    "batch_l": float(m.group(7)),
                    "policy_entropy": float(m.group(8)),
                    "value_out": float(m.group(9)),
                    "value_tar": float(m.group(10)),
                    "grad_norm": float(m.group(11)),
                    "lr": float(m.group(12)),
                })
    return rows


def smooth(values, window=100):
    """Simple moving average."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training(parent, step_dirs):
    """Generate training_metrics.png."""
    log_files = []
    for sd in step_dirs:
        lf = os.path.join(sd, "training_run.log")
        if os.path.exists(lf):
            log_files.append(lf)

    # Also check parent dir
    root_log = os.path.join(parent, "training_run.log")
    if os.path.exists(root_log):
        log_files.insert(0, root_log)

    if not log_files:
        print("  No training_run.log files found, skipping training_metrics.png")
        return

    all_rows = []
    for lf in log_files:
        rows = parse_training_log(lf)
        all_rows.extend(rows)
        print(f"  Training log {os.path.basename(os.path.dirname(lf))}: {len(rows)} steps")

    # Sort and deduplicate
    all_rows.sort(key=lambda r: r["step"])
    seen = set()
    deduped = []
    for r in all_rows:
        if r["step"] not in seen:
            seen.add(r["step"])
            deduped.append(r)
    all_rows = deduped

    if not all_rows:
        print("  No training data parsed, skipping training_metrics.png")
        return

    print(f"  Total: {len(all_rows)} steps ({all_rows[0]['step']} - {all_rows[-1]['step']})")

    steps = [r["step"] for r in all_rows]
    sw = 200

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Talbot Training Metrics", fontsize=15, fontweight="bold")

    # Total Loss
    ax = axes[0, 0]
    raw = [r["total_loss"] for r in all_rows]
    ax.plot(steps, raw, alpha=0.15, color="gray", linewidth=0.5)
    s = smooth(raw, sw)
    if len(s) > 0:
        ax.plot(steps[sw - 1:], s, color="#4ade80", linewidth=1.5, label=f"Smoothed ({sw})")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Policy vs Value Loss
    ax = axes[0, 1]
    p_raw = [r["policy_loss"] for r in all_rows]
    v_raw = [r["value_loss"] for r in all_rows]
    sp = smooth(p_raw, sw)
    sv = smooth(v_raw, sw)
    if len(sp) > 0:
        ax.plot(steps[sw - 1:], sp, color="#60a5fa", linewidth=1.5, label="Policy loss")
        ax.plot(steps[sw - 1:], sv, color="#f87171", linewidth=1.5, label="Value loss")
    ax.set_ylabel("Loss")
    ax.set_title("Policy vs Value Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Policy Entropy
    ax = axes[1, 0]
    raw = [r["policy_entropy"] for r in all_rows]
    ax.plot(steps, raw, alpha=0.15, color="gray", linewidth=0.5)
    s = smooth(raw, sw)
    if len(s) > 0:
        ax.plot(steps[sw - 1:], s, color="#facc15", linewidth=1.5, label=f"Smoothed ({sw})")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Batch W/D/L
    ax = axes[1, 1]
    sw2 = sw
    s_w = smooth([r["batch_w"] for r in all_rows], sw2)
    s_d = smooth([r["batch_d"] for r in all_rows], sw2)
    s_l = smooth([r["batch_l"] for r in all_rows], sw2)
    if len(s_w) > 0:
        ax.plot(steps[sw2 - 1:], s_w, color="#4ade80", linewidth=1.5, label="Win %")
        ax.plot(steps[sw2 - 1:], s_d, color="#facc15", linewidth=1.5, label="Draw %")
        ax.plot(steps[sw2 - 1:], s_l, color="#f87171", linewidth=1.5, label="Loss %")
    ax.set_ylabel("%")
    ax.set_title("Batch Win / Draw / Loss %")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Gradient Norm
    ax = axes[2, 0]
    raw = [r["grad_norm"] for r in all_rows]
    ax.plot(steps, raw, alpha=0.15, color="gray", linewidth=0.5)
    s = smooth(raw, sw)
    if len(s) > 0:
        ax.plot(steps[sw - 1:], s, color="#c084fc", linewidth=1.5, label=f"Smoothed ({sw})")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Grad Norm")
    ax.set_title("Gradient Norm")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Value Output vs Target
    ax = axes[2, 1]
    s_vo = smooth([r["value_out"] for r in all_rows], sw)
    s_vt = smooth([r["value_tar"] for r in all_rows], sw)
    if len(s_vo) > 0:
        ax.plot(steps[sw - 1:], s_vo, color="#60a5fa", linewidth=1.5, label="V output")
        ax.plot(steps[sw - 1:], s_vt, color="#fb923c", linewidth=1.5, label="V target")
    ax.axhline(y=0, color="white", alpha=0.3, linewidth=0.5, linestyle="--")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Value")
    ax.set_title("Value Output vs Target")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(parent, "training_metrics.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <rl_dir>")
        sys.exit(1)

    parent = sys.argv[1]
    if not os.path.isdir(parent):
        print(f"Error: {parent} is not a directory")
        sys.exit(1)

    step_dirs = sorted(glob.glob(os.path.join(parent, "run_step_*")))
    if not step_dirs:
        print(f"No run_step_* folders found in {parent}")
        sys.exit(1)

    print(f"Found {len(step_dirs)} run_step folders\n")

    print("=== Opening Diversity ===")
    plot_openings(parent, step_dirs)

    print("\n=== Training Metrics ===")
    plot_training(parent, step_dirs)

    print("\nDone.")


if __name__ == "__main__":
    main()