"""Full evaluation of the trained models (Task 2.3).

usage:  python -m dl.evaluate
- accuracy, macro-F1, per-class precision/recall/F1, confusion matrix (clean test set)
- robustness test sets made with the Task 1 tools (noise, blur, rotation, darker
  light, new background, everything combined)
- comparison: CNN v1 vs CNN v2 vs a classical baseline (HSV histogram + logistic regression)
- Grad-CAM examples
outputs: outputs/metrics.json, outputs/figures/{confusion_*,robustness_bar,gradcam_examples,errors_combined}.png
"""
from __future__ import annotations

import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import data_pipeline as dp
from .model import FruitClassifier, load_model, overlay_heatmap
from .train import plot_confusion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUT, "figures")


def _rows(path):
    return dp.load_manifest(path)


def corrupt(X, kind, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "clean":
        return X
    if kind == "gaussian_noise":
        return np.clip(X + rng.normal(0, 0.08, X.shape), 0, 1).astype(np.float32)
    if kind == "blur":
        return np.stack([cv2.GaussianBlur(x, (0, 0), 1.5) for x in X]).astype(np.float32)
    if kind == "rotate_45":
        M = cv2.getRotationMatrix2D((X.shape[2] / 2, X.shape[1] / 2), 45, 1.0)
        return np.stack([cv2.warpAffine(x, M, X.shape[1:3], borderValue=(1, 1, 1)) for x in X]).astype(np.float32)
    if kind == "darker":
        return (X * 0.6).astype(np.float32)
    if kind == "new_background":
        return dp.replace_background(X, seed=99)
    if kind == "combined":
        Y = dp.replace_background(X, seed=123) * 0.75
        Y = np.stack([cv2.GaussianBlur(y.astype(np.float32), (0, 0), 1.0) for y in Y])
        return np.clip(Y + rng.normal(0, 0.05, Y.shape), 0, 1).astype(np.float32)
    raise ValueError(kind)


KINDS = ["clean", "gaussian_noise", "blur", "rotate_45", "darker", "new_background", "combined"]


def hsv_features(X):
    out = []
    for x in X:
        hsv = cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]).ravel()
        out.append(h / (h.sum() + 1e-8))
    return np.array(out)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    train, test = _rows(os.path.join(OUT, "manifest_train.csv")), _rows(os.path.join(OUT, "manifest_test.csv"))
    Xtr, ytr = dp.load_arrays(train)
    Xte, yte = dp.load_arrays(test)
    sets = {k: corrupt(Xte, k, seed=7) for k in KINDS}

    models = {}
    for v in ("v1", "v2"):
        p = os.path.join(ROOT, "models", f"fruit_cnn_{v}.keras")
        if os.path.exists(p):
            models[f"CNN {v}"] = load_model(p)
    base = LogisticRegression(max_iter=3000, C=10).fit(hsv_features(Xtr), ytr)

    robust, preds = {}, {}
    for name, m in models.items():
        robust[name] = {}
        for k, X in sets.items():
            yh = m.predict(X, verbose=0).argmax(1)
            preds[(name, k)] = yh
            robust[name][k] = round(float((yh == yte).mean()), 4)
    robust["HSV-hist + LogReg"] = {k: round(float((base.predict(hsv_features(X)) == yte).mean()), 4)
                                   for k, X in sets.items()}
    print(json.dumps(robust, indent=1))

    # ---- detailed metrics for the deployed model (v2) ----
    dep = "CNN v2" if "CNN v2" in models else list(models)[0]
    metrics = {"deployed_model": dep, "robustness_accuracy": robust}
    for k in ("clean", "combined"):
        yh = preds[(dep, k)]
        cm = confusion_matrix(yte, yh, labels=range(len(dp.CLASSES)))
        rep = classification_report(yte, yh, target_names=dp.CLASSES, digits=4, output_dict=True, zero_division=0)
        metrics[k] = {"accuracy": float((yh == yte).mean()), "macro_f1": float(f1_score(yte, yh, average="macro")),
                      "per_class": {c: {kk: round(rep[c][kk], 4) for kk in ("precision", "recall", "f1-score", "support")}
                                    for c in dp.CLASSES},
                      "confusion_matrix": cm.tolist()}
        plot_confusion(cm, dp.CLASSES, os.path.join(FIG, f"confusion_{k}.png"),
                       f"{dep} – {k} test set (acc={metrics[k]['accuracy']:.3f})")
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- robustness bar chart ----
    names = list(robust)
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    colors = ["#9aa5b1", "#2a6fdb", "#e0a526"]
    for i, n in enumerate(names):
        vals = [robust[n][k] for k in KINDS]
        b = ax.bar(np.arange(len(KINDS)) + (i - (len(names) - 1) / 2) * w, vals, w, label=n, color=colors[i % 3])
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(KINDS))); ax.set_xticklabels(KINDS)
    ax.set_ylim(0, 1.1); ax.set_ylabel("test accuracy"); ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Robustness of the classifiers on modified versions of the test set"); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "robustness_bar.png"), dpi=100); plt.close()

    # ---- Grad-CAM examples (deployed model) ----
    clf = FruitClassifier(os.path.join(ROOT, "models"))
    rng = np.random.default_rng(5)
    pick = [int(rng.choice(np.nonzero(yte == c)[0])) for c in range(len(dp.CLASSES))]
    fig, ax = plt.subplots(2, 8, figsize=(17, 4.8))
    for j, i in enumerate(pick):
        src = sets["clean"][i] if j % 2 == 0 else sets["new_background"][i]
        bgr = cv2.cvtColor(np.uint8(src * 255), cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (128, 128), interpolation=cv2.INTER_CUBIC)
        top, _ = clf.predict(bgr)
        cam = clf.gradcam(bgr)
        ax[0, j].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); ax[0, j].axis("off")
        ax[0, j].set_title(dp.CLASSES[yte[i]], fontsize=8)
        ax[1, j].imshow(cv2.cvtColor(overlay_heatmap(bgr, cam), cv2.COLOR_BGR2RGB)); ax[1, j].axis("off")
        ax[1, j].set_title(f"{top[0][0]} {100 * top[0][1]:.0f}%", fontsize=8)
    plt.suptitle("Grad-CAM of the deployed CNN (top: input, bottom: prediction + heat-map)", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "gradcam_examples.png"), dpi=95); plt.close()

    # ---- some errors on the hardest set ----
    yh = preds[(dep, "combined")]
    wrong = np.nonzero(yh != yte)[0][:8]
    if len(wrong):
        fig, ax = plt.subplots(1, len(wrong), figsize=(2.1 * len(wrong), 2.6))
        ax = np.atleast_1d(ax)
        for a, i in zip(ax, wrong):
            a.imshow(sets["combined"][i]); a.axis("off")
            a.set_title(f"true: {dp.CLASSES[yte[i]]}\npred: {dp.CLASSES[yh[i]]}", fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "errors_combined.png"), dpi=95); plt.close()
    print("deployed:", dep, "clean acc", metrics["clean"]["accuracy"], "combined acc", metrics["combined"]["accuracy"])


if __name__ == "__main__":
    main()
