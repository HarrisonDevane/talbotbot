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

def parse_games(folder):
    """Parse all worker_*.log files in a folder for first moves and castling."""
    files = sorted(glob.glob(os.path.join(folder, "worker_*.log")))
    if not files:
        return None, None, 0, None

    white = defaultdict(int)
    black = defaultdict(int)
    total = 0
    # castling counts: w_king, w_queen, b_king, b_queen, w_none, b_none
    castling = {"w_king": 0, "w_queen": 0, "b_king": 0, "b_queen": 0, "w_none": 0, "b_none": 0}

    for f in files:
        with open(f) as fh:
            content = fh.read()
        for _, moves in re.findall(
            r'\[Result "([^"]+)"\].*?\n\n(1\. .+?)(?:\n\n|\Z)', content, re.DOTALL
        ):
            flat = moves.replace("\n", " ")
            m = re.match(r"1\. (\S+) (\S+)", flat)
            if m:
                white[m.group(1)] += 1
                black[m.group(2)] += 1
                total += 1

            # Extract all moves, split into white/black by move numbers
            # Moves look like: 1. e4 e5 2. Nf3 Nc6 ...
            all_tokens = re.sub(r'\d+\.', '', flat).split()
            w_moves = all_tokens[0::2]
            b_moves = all_tokens[1::2]

            w_castled = False
            for mv in w_moves:
                if mv == "O-O-O":
                    castling["w_queen"] += 1
                    w_castled = True
                    break
                elif mv == "O-O":
                    castling["w_king"] += 1
                    w_castled = True
                    break
            if not w_castled:
                castling["w_none"] += 1

            b_castled = False
            for mv in b_moves:
                if mv == "O-O-O":
                    castling["b_queen"] += 1
                    b_castled = True
                    break
                elif mv == "O-O":
                    castling["b_king"] += 1
                    b_castled = True
                    break
            if not b_castled:
                castling["b_none"] += 1

    return white, black, total, castling


def plot_openings(parent, step_dirs):
    """Generate opening_trends.png with castling data."""
    steps = []
    w_main, w_junk, b_princ, b_junk = [], [], [], []
    w_d4, w_e4, w_nf3, w_c4 = [], [], [], []
    b_d5, b_e5, b_c5, b_nf6 = [], [], [], []
    w_castle_k, w_castle_q, w_castle_none = [], [], []
    b_castle_k, b_castle_q, b_castle_none = [], [], []

    for sd in step_dirs:
        try:
            step = int(os.path.basename(sd).replace("run_step_", ""))
        except ValueError:
            continue

        white, black, total, castling = parse_games(sd)
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

        w_castle_k.append(castling["w_king"] / total * 100)
        w_castle_q.append(castling["w_queen"] / total * 100)
        w_castle_none.append(castling["w_none"] / total * 100)
        b_castle_k.append(castling["b_king"] / total * 100)
        b_castle_q.append(castling["b_queen"] / total * 100)
        b_castle_none.append(castling["b_none"] / total * 100)

        print(f"  Step {step:>6}: {total:>5} games | W-main {w_main[-1]:5.1f}% | W-castle {100 - w_castle_none[-1]:5.1f}%")

    if not steps:
        print("  No opening data found, skipping opening_trends.png")
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
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
    ax.set_ylabel("%")
    ax.set_title("White Individual Openings")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, b_d5, "o-", ms=3, label="d5")
    ax.plot(steps, b_e5, "s-", ms=3, label="e5")
    ax.plot(steps, b_c5, "^-", ms=3, label="c5")
    ax.plot(steps, b_nf6, "D-", ms=3, label="Nf6")
    ax.set_ylabel("%")
    ax.set_title("Black Individual Openings")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(steps, w_castle_k, "o-", ms=3, color="#4ade80", label="White O-O")
    ax.plot(steps, w_castle_q, "s-", ms=3, color="#60a5fa", label="White O-O-O")
    ax.plot(steps, w_castle_none, "^-", ms=3, color="#f87171", label="White no castle")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("%")
    ax.set_title("White Castling %")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(steps, b_castle_k, "o-", ms=3, color="#4ade80", label="Black O-O")
    ax.plot(steps, b_castle_q, "s-", ms=3, color="#60a5fa", label="Black O-O-O")
    ax.plot(steps, b_castle_none, "^-", ms=3, color="#f87171", label="Black no castle")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("%")
    ax.set_title("Black Castling %")
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
    """Parse a single training_run.log (categorical WDL format)."""
    rows = []
    with open(filepath) as f:
        for line in f:
            m = re.search(
                r"Step (\d+) \| Loss: T=([\d.]+) \(P=([\d.]+), V=([\d.]+)\) \| "
                r"Tar \(W/D/L\): ([\d.]+)% / ([\d.]+)% / ([\d.]+)% \| "
                r"Pred: ([\d.]+)% / ([\d.]+)% / ([\d.]+)% \| "
                r"P_Ent=([\d.]+) \| "
                r"Grad=([\d.]+) \| LR=([\d.]+)",
                line,
            )
            if m:
                rows.append({
                    "step": int(m.group(1)),
                    "total_loss": float(m.group(2)),
                    "policy_loss": float(m.group(3)),
                    "value_loss": float(m.group(4)),
                    "tar_w": float(m.group(5)),
                    "tar_d": float(m.group(6)),
                    "tar_l": float(m.group(7)),
                    "pred_w": float(m.group(8)),
                    "pred_d": float(m.group(9)),
                    "pred_l": float(m.group(10)),
                    "policy_entropy": float(m.group(11)),
                    "grad_norm": float(m.group(12)),
                    "lr": float(m.group(13)),
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

    # Value Head Calibration Error (|Pred - Target| per WDL category)
    ax = axes[1, 1]
    err_w = [abs(r["pred_w"] - r["tar_w"]) for r in all_rows]
    err_d = [abs(r["pred_d"] - r["tar_d"]) for r in all_rows]
    err_l = [abs(r["pred_l"] - r["tar_l"]) for r in all_rows]
    s_ew = smooth(err_w, sw)
    s_ed = smooth(err_d, sw)
    s_el = smooth(err_l, sw)
    if len(s_ew) > 0:
        x = steps[sw - 1:]
        ax.plot(x, s_ew, color="#4ade80", linewidth=1.5, label="Win error")
        ax.plot(x, s_ed, color="#facc15", linewidth=1.5, label="Draw error")
        ax.plot(x, s_el, color="#f87171", linewidth=1.5, label="Loss error")
    ax.set_ylabel("% pts")
    ax.set_title("Value Head Calibration Error (|Pred - Tar|)")
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

    # Value Head: Predicted vs Target WDL
    ax = axes[2, 1]
    s_pw = smooth([r["pred_w"] for r in all_rows], sw)
    s_pd = smooth([r["pred_d"] for r in all_rows], sw)
    s_pl = smooth([r["pred_l"] for r in all_rows], sw)
    s_tw = smooth([r["tar_w"] for r in all_rows], sw)
    s_td = smooth([r["tar_d"] for r in all_rows], sw)
    s_tl = smooth([r["tar_l"] for r in all_rows], sw)
    if len(s_pw) > 0:
        x = steps[sw - 1:]
        ax.plot(x, s_pw, color="#4ade80", linewidth=1.5, label="Pred W")
        ax.plot(x, s_tw, color="#4ade80", linewidth=1.0, linestyle="--", alpha=0.5, label="Tar W")
        ax.plot(x, s_pd, color="#facc15", linewidth=1.5, label="Pred D")
        ax.plot(x, s_td, color="#facc15", linewidth=1.0, linestyle="--", alpha=0.5, label="Tar D")
        ax.plot(x, s_pl, color="#f87171", linewidth=1.5, label="Pred L")
        ax.plot(x, s_tl, color="#f87171", linewidth=1.0, linestyle="--", alpha=0.5, label="Tar L")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("%")
    ax.set_title("Value Head: Predicted vs Target WDL")
    ax.legend(fontsize=7, ncol=2)
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