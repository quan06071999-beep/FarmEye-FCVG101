"""Data pipeline for the fruit / vegetable classifier (Task 2).

Steps
  1. ingestion      : scan  <root>/<split>/<class>/*.jpg  -> manifest (path, label)
  2. cleaning       : unreadable files, wrong size / channels, (almost) blank
                      images, exact duplicates (MD5)  -> removed + logged
  3. split          : stratified train / validation split of the Training folder,
                      the official Test folder is kept as an untouched test set
  4. transformation : resize to IMG_SIZE, scale to [0,1]
  5. augmentation   : (train only) flip, rotation, zoom, translation, contrast,
                      brightness, Gaussian noise  -> done on the fly in tf.data
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter

import cv2
import numpy as np

IMG_SIZE = 64
SEED = 42

# the 8 classes used in the MVP (organic farm sorting line)
CLASSES = ["Apple Red 1", "Banana", "Lemon", "Orange", "Pepper Green",
           "Potato White", "Tomato 1", "Tomato not Ripened"]


# --------------------------------------------------------------------------
# 1. ingestion
# --------------------------------------------------------------------------
def ingest(root: str, split: str, classes=CLASSES) -> list[dict]:
    rows = []
    for ci, c in enumerate(classes):
        d = os.path.join(root, split, c)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"missing class folder: {d}")
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({"path": os.path.join(d, f), "label": ci, "class": c})
    return rows


# --------------------------------------------------------------------------
# 2. cleaning
# --------------------------------------------------------------------------
def clean(rows: list[dict], min_std: float = 5.0, expected=(100, 100)) -> tuple[list[dict], dict]:
    """Remove bad samples and return (clean_rows, report)."""
    report = Counter()
    seen = {}
    keep = []
    for r in rows:
        img = cv2.imread(r["path"], cv2.IMREAD_UNCHANGED)
        if img is None:
            report["unreadable"] += 1
            continue
        if img.ndim == 2 or img.shape[2] not in (3, 4):
            report["not_colour"] += 1               # could be converted, but rare -> drop
            continue
        if img.shape[:2] != expected:
            report["resized_wrong_size"] += 1       # kept: it will be resized later
        if float(img.std()) < min_std:
            report["blank"] += 1
            continue
        h = hashlib.md5(img.tobytes()).hexdigest()
        if h in seen:
            report["duplicate"] += 1
            continue
        seen[h] = r["path"]
        keep.append(r)
    report["kept"] = len(keep)
    report["input"] = len(rows)
    return keep, dict(report)


# --------------------------------------------------------------------------
# 3. split
# --------------------------------------------------------------------------
def stratified_split(rows: list[dict], val_frac: float = 0.15, seed: int = SEED):
    rng = np.random.default_rng(seed)
    train, val = [], []
    labels = sorted({r["label"] for r in rows})
    for l in labels:
        rs = [r for r in rows if r["label"] == l]
        idx = rng.permutation(len(rs))
        n_val = int(round(val_frac * len(rs)))
        val += [rs[i] for i in idx[:n_val]]
        train += [rs[i] for i in idx[n_val:]]
    return train, val


def class_weights(rows: list[dict], n_classes: int = len(CLASSES)) -> dict:
    """weight_c = N / (K * n_c)  -> balances the bigger 'Tomato 1' class."""
    cnt = Counter(r["label"] for r in rows)
    n = len(rows)
    return {c: n / (n_classes * cnt[c]) for c in range(n_classes)}


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_manifest(rows, path):
    """Paths are stored relative to the project folder (portable Windows <-> Linux)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "class"])
        w.writeheader()
        for r in rows:
            rel = os.path.relpath(os.path.abspath(r["path"]), PROJECT_ROOT).replace("\\", "/")
            w.writerow({**r, "path": rel})


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return [{"path": os.path.join(PROJECT_ROOT, *r["path"].split("/")), "label": int(r["label"]),
                 "class": r["class"]} for r in csv.DictReader(f)]


# --------------------------------------------------------------------------
# 4. transformation (shared by training, evaluation and the app!)
# --------------------------------------------------------------------------
def preprocess_bgr(img_bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """BGR uint8 (any size) -> RGB float32 (size, size, 3) in [0,1].

    The app uses exactly this function so train and inference match.
    """
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    if img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    # pad to a square with white (like the dataset background) so the aspect is kept
    h, w = img_bgr.shape[:2]
    s = max(h, w)
    canvas = np.full((s, s, 3), 255, np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img_bgr
    rgb = cv2.cvtColor(cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def load_arrays(rows: list[dict], size: int = IMG_SIZE):
    X = np.stack([preprocess_bgr(cv2.imread(r["path"]), size) for r in rows])
    y = np.array([r["label"] for r in rows], np.int32)
    return X, y


# --------------------------------------------------------------------------
# 4b. background replacement (feature engineering / augmentation, v2)
# --------------------------------------------------------------------------
def background_mask(img01: np.ndarray, thr: float = 0.82) -> np.ndarray:
    """True where the pixel belongs to the white studio background.

    Near-white pixels are only background if they are connected to the image
    border (so a white highlight on a lemon is not removed). Done with a flood
    fill from the 4 corners, then the background is grown by 1 px to remove the
    light anti-aliased halo around the fruit.
    """
    u8 = np.uint8(np.clip(img01, 0, 1) * 255)
    white = (u8.min(axis=2) >= int(thr * 255)).astype(np.uint8)
    h, w = white.shape
    ff = white.copy()
    seed_mask = np.zeros((h + 2, w + 2), np.uint8)
    for (x, y) in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if ff[y, x] == 1:
            cv2.floodFill(ff, seed_mask, (x, y), 2)
    bg = (ff == 2).astype(np.uint8)
    return cv2.dilate(bg, np.ones((3, 3), np.uint8)).astype(bool)


def random_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Random 'farm' background: solid colour, gradient (conveyor belt),
    smooth noise (soil / wood) or a crate-like stripe pattern. RGB float [0,1]."""
    kind = rng.integers(4)
    c1, c2 = rng.uniform(0.05, 0.95, 3), rng.uniform(0.05, 0.95, 3)
    if kind == 0:
        bg = np.ones((h, w, 3)) * c1
    elif kind == 1:
        t = np.linspace(0, 1, w)[None, :, None] if rng.random() < .5 else np.linspace(0, 1, h)[:, None, None]
        bg = c1 * (1 - t) + c2 * t
        bg = np.broadcast_to(bg, (h, w, 3))
    elif kind == 2:
        n = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), rng.uniform(1, 4))
        n = (n - n.min()) / (n.max() - n.min() + 1e-8)
        bg = c1 * (1 - n[..., None]) + c2 * n[..., None]
    else:
        period = rng.integers(6, 20)
        stripes = ((np.arange(w) // period) % 2).astype(np.float64)[None, :, None]
        bg = np.broadcast_to(c1 * (1 - stripes) + c2 * stripes, (h, w, 3))
    return np.clip(bg + rng.normal(0, 0.02, (h, w, 3)), 0, 1).astype(np.float32)


def replace_background(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Paste every fruit of X (N,H,W,3 in [0,1]) on a random background."""
    rng = np.random.default_rng(seed)
    out = X.copy()
    for i, x in enumerate(X):
        m = background_mask(x)
        out[i][m] = random_background(*x.shape[:2], rng)[m]
    return out


# --------------------------------------------------------------------------
# 5. augmentation + tf.data
# --------------------------------------------------------------------------
def build_augmenter(version: str = "v2"):
    """v1: geometric + mild photometric augmentation.
    v2: + stronger contrast, reflect padding (no white corners on coloured backgrounds),
        more noise; the multiplicative exposure gain is added in make_dataset."""
    from tensorflow import keras
    strong = version == "v2"
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.25, fill_mode="reflect" if strong else "constant", fill_value=1.0),
        keras.layers.RandomZoom((-0.2, 0.2), fill_mode="reflect" if strong else "constant", fill_value=1.0),
        keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect" if strong else "constant", fill_value=1.0),
        keras.layers.RandomContrast(0.35 if strong else 0.2),
        keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),  # additive shift (v2 also uses a gain)
        keras.layers.GaussianNoise(0.04 if strong else 0.03),   # sensor noise (training only)
    ], name="augmentation")


def make_dataset(X, y, batch: int = 64, training: bool = False, augmenter=None, gain_range=None):
    """tf.data pipeline. gain_range=(lo, hi): multiply every training image by a
    random exposure gain (Keras' RandomBrightness only ADDS a constant, but a
    darker room or evening light scales the intensities)."""
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch)
    if training and augmenter is not None:
        def _aug(a, b):
            a = augmenter(a, training=True)
            if gain_range is not None:
                g = tf.random.uniform([tf.shape(a)[0], 1, 1, 1], gain_range[0], gain_range[1])
                a = a * g
            return tf.clip_by_value(a, 0.0, 1.0), b
        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def run_pipeline(root: str, out_dir: str, val_frac: float = 0.15):
    """Run steps 1-3 and save manifests + a json report. Returns (train, val, test, report)."""
    os.makedirs(out_dir, exist_ok=True)
    train_raw = ingest(root, "Training")
    test_raw = ingest(root, "Test")
    train_clean, rep_tr = clean(train_raw)
    test_clean, rep_te = clean(test_raw)
    train, val = stratified_split(train_clean, val_frac)
    report = {
        "classes": CLASSES,
        "cleaning_training_folder": rep_tr,
        "cleaning_test_folder": rep_te,
        "n_train": len(train), "n_val": len(val), "n_test": len(test_clean),
        "per_class_train": {CLASSES[k]: v for k, v in sorted(Counter(r["label"] for r in train).items())},
        "per_class_val": {CLASSES[k]: v for k, v in sorted(Counter(r["label"] for r in val).items())},
        "per_class_test": {CLASSES[k]: v for k, v in sorted(Counter(r["label"] for r in test_clean).items())},
        "class_weights": {CLASSES[k]: round(v, 3) for k, v in class_weights(train).items()},
    }
    for name, rows in (("train", train), ("val", val), ("test", test_clean)):
        save_manifest(rows, os.path.join(out_dir, f"manifest_{name}.csv"))
    with open(os.path.join(out_dir, "pipeline_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return train, val, test_clean, report
