"""Plot train/val accuracy from openai/grok CSV logs and report steps-to-threshold.

Usage:
    python tools/plot_metrics.py RUN_DIR [RUN_DIR ...] [--out fig.png] [--thr 99]

RUN_DIR is the logdir passed to scripts/train.py; the script looks for
RUN_DIR/default/version_*/metrics.csv (the latest version).
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "default", "version_*", "metrics.csv")))
    if not files:
        raise FileNotFoundError(f"no metrics.csv under {run_dir}")
    d = pd.read_csv(files[-1])
    # step logged by Lightning is 0-based; +1 = number of optimizer updates done
    d["updates"] = d["step"] + 1
    train = d.dropna(subset=["train_accuracy"])[["updates", "train_accuracy"]]
    val = d.dropna(subset=["val_accuracy"])[["updates", "val_accuracy", "full_train_acc"]]
    return train, val


def first_reach(df, col, thr):
    hit = df[df[col] >= thr]
    return int(hit["updates"].iloc[0]) if len(hit) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="accuracy.png")
    ap.add_argument("--thr", type=float, default=99.0)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4.2))
    print(f"{'run':40s} {'train>=thr':>11s} {'val>=thr':>10s} {'ratio':>7s} {'final val':>9s}")
    for run in args.runs:
        train, val = load(run)
        name = os.path.basename(os.path.normpath(run))
        # full_train_acc is evaluated on the whole train set at val time (cleaner than
        # the running mean over minibatches in train_accuracy)
        t_hit = first_reach(val, "full_train_acc", args.thr)
        v_hit = first_reach(val, "val_accuracy", args.thr)
        ratio = f"{v_hit / t_hit:.0f}x" if (t_hit and v_hit) else "-"
        print(f"{name:40s} {str(t_hit):>11s} {str(v_hit):>10s} {ratio:>7s} {val['val_accuracy'].iloc[-1]:9.2f}")
        line, = ax.plot(val["updates"], val["full_train_acc"], ls="--", lw=1.2)
        ax.plot(val["updates"], val["val_accuracy"], color=line.get_color(), lw=1.8, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Optimization steps")
    ax.set_ylabel("Accuracy, %")
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.3)
    ax.legend(title="solid = val, dashed = train", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
