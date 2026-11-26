"""Train + evaluate the CNN and save the model artifact (Task 2).

usage:  python -m dl.train --version v2 --epochs 25
   v1 = original white-background images + mild augmentation
   v2 = v1 + background replacement + stronger photometric augmentation (deployed)
outputs: models/fruit_cnn_<version>.keras (v2 also -> models/fruit_cnn.keras),
         models/labels.json, outputs/train_<version>.json, outputs/figures/*.png
then run  python -m dl.evaluate  for the full evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import data_pipeline as dp
from .model import build_cnn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_history(h, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="val")
    ax[0].set_title("loss (cross-entropy)"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(h["accuracy"], label="train"); ax[1].plot(h["val_accuracy"], label="val")
    ax[1].set_title("accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def plot_confusion(cm, classes, path, title):
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=8); ax.set_yticklabels(classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    plt.colorbar(im, fraction=0.046); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "fruits"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--version", choices=["v1", "v2"], default="v2")
    ap.add_argument("--finetune", default=None,
                    help="path of a saved model to continue training from (stage 2, small lr)")
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(dp.SEED); np.random.seed(dp.SEED)

    out_dir = os.path.join(ROOT, "outputs"); fig_dir = os.path.join(out_dir, "figures")
    model_dir = os.path.join(ROOT, "models")
    for d in (out_dir, fig_dir, model_dir):
        os.makedirs(d, exist_ok=True)

    # ---------------- data pipeline ----------------
    train, val, test, report = dp.run_pipeline(args.data, out_dir)
    print(json.dumps(report, indent=1))
    Xtr, ytr = dp.load_arrays(train); Xva, yva = dp.load_arrays(val); Xte, yte = dp.load_arrays(test)
    if args.version == "v2":
        # every training / validation image also appears once on a random background
        Xtr = np.concatenate([Xtr, dp.replace_background(Xtr, seed=1)]); ytr = np.concatenate([ytr, ytr])
        Xva = np.concatenate([Xva, dp.replace_background(Xva, seed=2)]); yva = np.concatenate([yva, yva])
    print("train", Xtr.shape, "val", Xva.shape, "test", Xte.shape)
    aug = dp.build_augmenter(args.version)
    gain = (0.45, 1.15) if args.version == "v2" else None
    ds_tr = dp.make_dataset(Xtr, ytr, args.batch, training=True, augmenter=aug, gain_range=gain)
    ds_va = dp.make_dataset(Xva, yva, args.batch)

    # figure: a few augmented samples
    sample = Xtr[np.random.default_rng(1).choice(len(Xtr), 6, replace=False)]
    fig, ax = plt.subplots(2, 6, figsize=(12, 4.3))
    for i in range(6):
        ax[0, i].imshow(sample[i]); ax[0, i].axis("off"); ax[0, i].set_title("original", fontsize=9)
        a = aug(sample[i:i + 1], training=True)[0].numpy() * (np.random.uniform(*gain) if gain else 1.0)
        ax[1, i].imshow(np.clip(a, 0, 1)); ax[1, i].axis("off")
        ax[1, i].set_title("augmented", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, f"augmentation_examples_{args.version}.png"), dpi=100); plt.close()

    # ---------------- model ----------------
    if args.finetune:
        from .model import load_model
        model = load_model(args.finetune)          # stage 2: continue from stage-1 weights
    else:
        model = build_cnn(len(dp.CLASSES), dp.IMG_SIZE, standardize=(args.version == "v2"))
    model.compile(optimizer=keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)]
    t0 = time.time()
    hist = model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, callbacks=cbs,
                     class_weight=dp.class_weights(train), verbose=2)
    train_time = time.time() - t0
    plot_history(hist.history, os.path.join(fig_dir, f"training_curves_{args.version}.png"))

    # ---------------- quick test-set check (full evaluation in dl/evaluate.py) ----------------
    yhat = model.predict(Xte, verbose=0).argmax(1)
    info = {"version": args.version, "test_accuracy_clean": float((yhat == yte).mean()),
            "val_accuracy_best": float(max(hist.history["val_accuracy"])),
            "epochs_run": len(hist.history["loss"]), "train_time_s": round(train_time, 1),
            "n_train_images": int(len(Xtr)), "n_val_images": int(len(Xva)),
            "params": int(model.count_params()), "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
            "hyperparameters": {"img_size": dp.IMG_SIZE, "batch": args.batch, "lr": args.lr,
                                "optimizer": "Adam", "loss": "sparse categorical cross-entropy",
                                "max_epochs": args.epochs, "early_stopping_patience": 8, "exposure_gain": list(gain) if gain else None,
                                "reduce_lr_patience": 3, "dropout": 0.4, "val_frac": 0.15,
                                "batchnorm_momentum": 0.9,
                                "per_image_standardization": args.version == "v2",
                                "finetuned_from": args.finetune}}
    with open(os.path.join(out_dir, f"train_{args.version}.json"), "w") as f:
        json.dump(info, f, indent=2)

    # ---------------- save the artifact for Task 3 ----------------
    model.save(os.path.join(model_dir, f"fruit_cnn_{args.version}.keras"))
    if args.version == "v2":
        model.save(os.path.join(model_dir, "fruit_cnn.keras"))       # the deployed model
    with open(os.path.join(model_dir, "labels.json"), "w") as f:
        json.dump({"classes": dp.CLASSES, "img_size": dp.IMG_SIZE}, f, indent=2)
    print(json.dumps({k: v for k, v in info.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()
