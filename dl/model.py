"""CNN model (VGG-style) + Grad-CAM + inference helper (Task 2)."""
from __future__ import annotations

import json
import os

import numpy as np


def _std_layer_class():
    """Per-image standardisation  x' = (x - mean(x)) / std(x)  as a saveable Keras layer.

    Same idea as NCC in template matching: the output does not change when the
    whole image is multiplied by a gain (darker / brighter light) or shifted by
    a constant, so the CNN does not have to learn this invariance from data.
    """
    import keras
    import tensorflow as tf

    @keras.saving.register_keras_serializable(package="farmeye")
    class PerImageStandardization(keras.layers.Layer):
        def call(self, x):
            mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
            std = tf.math.reduce_std(x, axis=[1, 2, 3], keepdims=True)
            return (x - mean) / tf.maximum(std, 1e-3)

    return PerImageStandardization


PerImageStandardization = None


def get_custom_objects():
    global PerImageStandardization
    if PerImageStandardization is None:
        PerImageStandardization = _std_layer_class()
    return {"farmeye>PerImageStandardization": PerImageStandardization,
            "PerImageStandardization": PerImageStandardization}


def load_model(path):
    from tensorflow import keras
    return keras.models.load_model(path, custom_objects=get_custom_objects())


def build_cnn(n_classes: int, img_size: int = 64, dropout: float = 0.4, standardize: bool = True):
    """Small VGG-style network:
       [per-image standardisation] ->
       [Conv32-BN-ReLU x2 -> MaxPool] -> [Conv64 x2 -> MaxPool] -> [Conv128 x2 -> MaxPool]
       -> GlobalAveragePooling -> Dropout -> Dense(softmax)
    ~290k parameters, trains on CPU in about 15 minutes.
    (v1 was trained with standardize=False.)
    """
    from tensorflow import keras
    L = keras.layers

    def block(x, f, name):
        for i in range(2):
            x = L.Conv2D(f, 3, padding="same", use_bias=False, name=f"{name}_conv{i + 1}")(x)
            x = L.BatchNormalization(momentum=0.9, name=f"{name}_bn{i + 1}")(x)   # faster running stats (small dataset)
            x = L.ReLU(name=f"{name}_relu{i + 1}")(x)
        return L.MaxPooling2D(2, name=f"{name}_pool")(x)

    inp = keras.Input((img_size, img_size, 3), name="image")
    x = get_custom_objects()["PerImageStandardization"](name="standardize")(inp) if standardize else inp
    x = block(x, 32, "b1")
    x = block(x, 64, "b2")
    x = block(x, 128, "b3")          # last conv layer -> used by Grad-CAM
    x = L.GlobalAveragePooling2D(name="gap")(x)
    x = L.Dropout(dropout, name="dropout")(x)
    out = L.Dense(n_classes, activation="softmax", name="probs")(x)
    return keras.Model(inp, out, name="farmeye_cnn")


LAST_CONV = "b3_relu2"


class FruitClassifier:
    """Loads the saved model artifact and runs predictions (+ Grad-CAM)."""

    def __init__(self, model_dir: str):
        self.model = load_model(os.path.join(model_dir, "fruit_cnn.keras"))
        with open(os.path.join(model_dir, "labels.json")) as f:
            meta = json.load(f)
        self.classes = meta["classes"]
        self.img_size = meta["img_size"]

    def predict(self, img_bgr: np.ndarray, top_k: int = 3):
        from .data_pipeline import preprocess_bgr
        x = preprocess_bgr(img_bgr, self.img_size)[None]
        p = self.model.predict(x, verbose=0)[0]
        order = np.argsort(-p)[:top_k]
        return [(self.classes[i], float(p[i])) for i in order], p

    def gradcam(self, img_bgr: np.ndarray, class_idx: int | None = None) -> np.ndarray:
        """Grad-CAM heat-map (H, W) in [0,1] at the input image resolution."""
        import cv2
        import tensorflow as tf
        from .data_pipeline import preprocess_bgr
        x = tf.convert_to_tensor(preprocess_bgr(img_bgr, self.img_size)[None])
        grad_model = tf.keras.Model(self.model.inputs,
                                    [self.model.get_layer(LAST_CONV).output, self.model.output])
        with tf.GradientTape() as tape:
            conv, preds = grad_model(x, training=False)
            if class_idx is None:
                class_idx = int(tf.argmax(preds[0]))
            score = preds[:, class_idx]
        grads = tape.gradient(score, conv)[0]                 # (h, w, C)
        weights = tf.reduce_mean(grads, axis=(0, 1))          # alpha_k = GAP of gradients
        cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
        cam = cam / (cam.max() + 1e-8)
        # preprocess pads to a square -> map back to the square then crop the padding
        h, w = img_bgr.shape[:2]
        s = max(h, w)
        cam_sq = cv2.resize(cam, (s, s), interpolation=cv2.INTER_LINEAR)
        y0, x0 = (s - h) // 2, (s - w) // 2
        return cam_sq[y0:y0 + h, x0:x0 + w]


def overlay_heatmap(img_bgr, cam, alpha: float = 0.45):
    import cv2
    hm = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, hm, alpha, 0)
