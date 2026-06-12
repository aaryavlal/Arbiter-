"""
audit_color_distributions.py
-----------------------------
Detects whether AWB-induced capture-process differences created a
class-correlated color confound between data/raw/paper/ and data/raw/trash/.

Only Pi-captured images are analyzed, identified by filename prefix:
  paper_*.jpg  in data/raw/paper/
  trash_*.jpg  in data/raw/trash/
TrashNet originals (numeric filenames like 1234.jpg) are explicitly excluded
because the AWB confound only applies to images captured through the Pi
camera pipeline.

Channel order note: picamera2 saves images to disk as BGR (OpenCV native
order). All per-channel statistics below use that order:
  index 0 → Blue
  index 1 → Green
  index 2 → Red
Variables are named _r/_g/_b for readability; they map to cv2 channel indices
2, 1, 0 respectively.
"""

import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = PROJECT_ROOT / "data" / "raw" / "paper"
TRASH_DIR = PROJECT_ROOT / "data" / "raw" / "trash"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "audit_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORNER_SIZE = 80   # pixels — side length of each corner patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_images(folder: Path, prefix: str, label: str):
    """
    Load Pi-captured .jpg files from folder matching {prefix}_*.jpg.
    Returns a list of (filename, image_ndarray) tuples.
    Skips unreadable files and logs the count.
    Also warns about dimension inconsistency.
    """
    all_jpg = sorted(folder.glob("*.jpg"))
    pi_paths = sorted(folder.glob(f"{prefix}_*.jpg"))
    skipped_filter = len(all_jpg) - len(pi_paths)

    images = []
    skipped_read = 0
    shapes = set()

    for p in pi_paths:
        img = cv2.imread(str(p))
        if img is None:
            skipped_read += 1
            continue
        shapes.add(img.shape[:2])  # (H, W)
        images.append((p.name, img))

    if skipped_read:
        print(f"  [{label}] {skipped_read} file(s) failed to load with cv2.imread.")
    if len(images) < 20:
        print(f"  WARNING: fewer than 20 images in '{label}' — statistics will be noisy.")
    if len(shapes) > 1:
        print(f"  WARNING: inconsistent image dimensions detected in '{label}': {shapes}")
        print("           Dimension variation is itself a capture-consistency issue.")

    return images, skipped_filter


def corner_patch_mean(img):
    """
    Sample four 80×80 patches at the image corners (native resolution).
    Returns mean_r, mean_g, mean_b over the union of all four patches.
    BGR channel order: index 0=B, 1=G, 2=R.
    """
    h, w = img.shape[:2]
    s = CORNER_SIZE
    patches = [
        img[0:s,   0:s,   :],   # top-left
        img[0:s,   w-s:w, :],   # top-right
        img[h-s:h, 0:s,   :],   # bottom-left
        img[h-s:h, w-s:w, :],   # bottom-right
    ]
    combined = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    # combined columns: [B, G, R] (cv2 BGR)
    mean_b = combined[:, 0].mean()
    mean_g = combined[:, 1].mean()
    mean_r = combined[:, 2].mean()
    return mean_r, mean_g, mean_b


def compute_features(filename: str, img, label: str) -> dict:
    """Compute all color features for a single image."""
    # Full-frame per-channel means — BGR layout, names reflect semantic channel
    mean_b = img[:, :, 0].mean()  # cv2 channel 0 = Blue
    mean_g = img[:, :, 1].mean()  # cv2 channel 1 = Green
    mean_r = img[:, :, 2].mean()  # cv2 channel 2 = Red

    eps = 1e-6  # avoid division by zero
    rg_ratio = mean_r / (mean_g + eps)
    bg_ratio  = mean_b / (mean_g + eps)

    corner_r, corner_g, corner_b = corner_patch_mean(img)
    corner_rg_ratio = corner_r / (corner_g + eps)
    corner_bg_ratio = corner_b / (corner_g + eps)

    return {
        "filename": filename,
        "class": label,
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "rg_ratio": rg_ratio,
        "bg_ratio": bg_ratio,
        "corner_r": corner_r,
        "corner_g": corner_g,
        "corner_b": corner_b,
        "corner_rg_ratio": corner_rg_ratio,
        "corner_bg_ratio": corner_bg_ratio,
    }


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d = (mean_a - mean_b) / pooled_std."""
    n_a, n_b = len(a), len(b)
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled_std


def print_stats_table(df: pd.DataFrame, features: list):
    """Print a table of per-feature stats sorted by |Cohen's d|."""
    paper = df[df["class"] == "paper"]
    trash = df[df["class"] == "trash"]

    rows = []
    for feat in features:
        p_vals = paper[feat].values
        t_vals = trash[feat].values
        d = cohen_d(p_vals, t_vals)
        rows.append({
            "feature": feat,
            "paper": f"{p_vals.mean():.3f} ± {p_vals.std():.3f}",
            "trash": f"{t_vals.mean():.3f} ± {t_vals.std():.3f}",
            "cohen_d": d,
        })

    rows.sort(key=lambda r: abs(r["cohen_d"]), reverse=True)

    col_w = [max(len(r["feature"]) for r in rows) + 2, 22, 22, 10]
    header = (
        f"{'feature':<{col_w[0]}}"
        f"{'paper_mean ± std':>{col_w[1]}}"
        f"{'trash_mean ± std':>{col_w[2]}}"
        f"{'cohen_d':>{col_w[3]}}"
    )
    print("\n" + header)
    print("-" * sum(col_w))
    for r in rows:
        print(
            f"{r['feature']:<{col_w[0]}}"
            f"{r['paper']:>{col_w[1]}}"
            f"{r['trash']:>{col_w[2]}}"
            f"{r['cohen_d']:>{col_w[3]}.3f}"
        )
    print()
    return rows


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

CLASS_COLORS = {"paper": "#2196F3", "trash": "#FF5722"}


def plot_histogram(df: pd.DataFrame, feature: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color in CLASS_COLORS.items():
        vals = df[df["class"] == label][feature].values
        ax.hist(vals, bins=40, alpha=0.55, color=color, label=label, density=True)
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {feature} by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_scatter_corner_ratios(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, color in CLASS_COLORS.items():
        sub = df[df["class"] == label]
        ax.scatter(
            sub["corner_rg_ratio"],
            sub["corner_bg_ratio"],
            c=color, alpha=0.4, s=12, label=label, edgecolors="none"
        )
    ax.set_xlabel("corner_rg_ratio  (R/G in corner patches)")
    ax.set_ylabel("corner_bg_ratio  (B/G in corner patches)")
    ax.set_title("Corner AWB ratios by class\n(distinct clusters → capture conditions differed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("AWB / Color Distribution Audit  (Pi-captured images only)")
    print("=" * 60)

    # 1. Load Pi-captured images only (filename prefix filter)
    print("\nLoading Pi-captured images only (filename prefix filter):")
    paper_imgs, paper_skipped = load_images(PAPER_DIR, "paper", "paper")
    trash_imgs, trash_skipped = load_images(TRASH_DIR, "trash", "trash")
    print(f"  {PAPER_DIR}/paper_*.jpg → {len(paper_imgs)} files")
    print(f"  {TRASH_DIR}/trash_*.jpg → {len(trash_imgs)} files")
    print(f"  (Skipped {paper_skipped} files in paper/ and "
          f"{trash_skipped} files in trash/ that did not match the prefix.)")

    if len(paper_imgs) < 10 or len(trash_imgs) < 10:
        sys.exit("ERROR: fewer than 10 Pi-captured images in at least one class. "
                 "Statistics won't be meaningful. Aborting.")

    if not paper_imgs and not trash_imgs:
        sys.exit("No images loaded from either folder. Aborting.")

    # 2. Compute features
    print("\nComputing per-image features…")
    rows = []
    for filename, img in paper_imgs:
        rows.append(compute_features(filename, img, "paper"))
    for filename, img in trash_imgs:
        rows.append(compute_features(filename, img, "trash"))

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = OUTPUT_DIR / "per_image_features.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"  Saved: {csv_path}")

    # 3. Stats table
    STAT_FEATURES = [
        "mean_r", "mean_g", "mean_b",
        "rg_ratio", "bg_ratio",
        "corner_r", "corner_g", "corner_b",
        "corner_rg_ratio", "corner_bg_ratio",
    ]
    print("\n--- Per-feature statistics (sorted by |Cohen's d|) ---")
    stat_rows = print_stats_table(df, STAT_FEATURES)

    # 4. Logistic regression shortcut detection
    LOGREG_FEATURES = [
        "mean_r", "mean_g", "mean_b",
        "rg_ratio", "bg_ratio",
        "corner_rg_ratio", "corner_bg_ratio",
    ]
    X = df[LOGREG_FEATURES].values
    y = (df["class"] == "paper").astype(int).values  # paper=1, trash=0

    # Class-prior baseline accuracy
    prior_acc = max(y.mean(), 1 - y.mean())

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    logreg_acc = cv_scores.mean()

    print(f"--- Logistic Regression (5-fold stratified CV) ---")
    print(f"  Class-prior baseline accuracy : {prior_acc:.3f}")
    print(f"  CV accuracy (mean ± std)      : {logreg_acc:.3f} ± {cv_scores.std():.3f}")
    print(f"  Individual folds              : {[f'{s:.3f}' for s in cv_scores]}")
    print()

    # 5. Visualizations
    print("--- Saving visualizations ---")
    plot_histogram(df, "rg_ratio",        OUTPUT_DIR / "histogram_rg_ratio.png")
    plot_histogram(df, "bg_ratio",        OUTPUT_DIR / "histogram_bg_ratio.png")
    plot_scatter_corner_ratios(df,        OUTPUT_DIR / "scatter_corner_ratios.png")

    # 6. Verdict
    corner_ratio_feats = ["corner_rg_ratio", "corner_bg_ratio"]
    paper_df = df[df["class"] == "paper"]
    trash_df = df[df["class"] == "trash"]

    max_corner_d = max(
        abs(cohen_d(paper_df[f].values, trash_df[f].values))
        for f in corner_ratio_feats
    )
    worst_corner_feat = max(
        corner_ratio_feats,
        key=lambda f: abs(cohen_d(paper_df[f].values, trash_df[f].values))
    )

    print("=" * 60)
    print("VERDICT")
    print("=" * 60)

    triggered = False

    if max_corner_d > 0.8:
        triggered = True
        print(
            f"STRONG SIGNAL: Pi-captured backgrounds differ substantially between classes "
            f"(corner Cohen's d = {max_corner_d:.3f} on {worst_corner_feat}). "
            f"Capture conditions were not consistent for the Pi-captured subset. "
            f"Recommend recapture."
        )

    if logreg_acc > 0.80:
        triggered = True
        print(
            f"Color statistics alone can separate Pi-captured classes "
            f"(CV accuracy = {logreg_acc:.3f}). "
            f"Model will likely shortcut on this in the Pi-captured subset. "
            f"Recommend recapture or audit."
        )

    if not triggered:
        print(
            f"No strong evidence of capture-process confound in the Pi-captured subset "
            f"(max corner Cohen's d = {max_corner_d:.3f}, "
            f"logistic regression CV accuracy = {logreg_acc:.3f}). "
            f"Real object color differences may still exist (expected); "
            f"proceed but verify on held-out test data."
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
