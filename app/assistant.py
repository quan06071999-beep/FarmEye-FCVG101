"""FarmEye assistant: intent recognition + tool routing (Task 3).

The user types a request in natural language (English or Vietnamese), e.g.
    "is this tomato ripe?"          -> classify
    "khử nhiễu ảnh bằng median 7"    -> filter(method=median, ksize=7)
    "match these two photos with ORB, ratio 0.8" -> features(method=orb, ratio=0.8)
The router scores every intent with weighted keywords, extracts parameters with
regular expressions, and then calls the matching tool (Task 1 engine or the
Task 2 CNN). Every tool returns a ToolResult (text + table + images) that the
UI shows to the user.

CLI:  python -m app.assistant "classify this fruit" --image data/samples/apple.jpg
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cv_engine import calibration as C, filtering as F, features as FT, measure as M  # noqa: E402

# ---------------------------------------------------------------------------
# 1) intent recognition
# ---------------------------------------------------------------------------
INTENTS = {
    "calibrate": {"calibrat": 3, "undistort": 3, "distortion": 2, "intrinsic": 3, "checkerboard": 3,
                  "chessboard": 3, "camera matrix": 3, "lens": 1, "hieu chinh": 3, "hieu chuan": 3,
                  "meo": 2, "ban co": 2, "noi tai": 2},
    "filter": {"denois": 3, "noise": 2, "filter": 2, "smooth": 2, "blur": 2, "median": 3, "gaussian": 2,
               "box": 2, "clean": 1, "nhieu": 3, "loc": 2, "lam min": 2, "lam mo": 2},
    "edges": {"edge": 3, "gradient": 3, "sobel": 3, "canny": 3, "contour": 1, "bien": 3, "canh": 2,
              "duong vien": 2, "dao ham": 2},
    "frequency": {"frequency": 3, "fourier": 3, "fft": 3, "spectrum": 3, "low-pass": 2, "lowpass": 2,
                  "high-pass": 2, "highpass": 2, "tan so": 3, "pho": 2},
    "features": {"feature": 3, "keypoint": 3, "interest point": 3, "corner": 2, "harris": 3, "sift": 3,
                 "orb": 3, "match": 3, "same object": 2, "the same": 2, "reference": 2, "compare two": 2, "dac trung": 3, "diem": 1,
                 "ghep": 2, "khop": 2, "so khop": 3, "goc": 1},
    "classify": {"classif": 3, "what fruit": 3, "what is this": 3, "which fruit": 3, "recogni": 2,
                 "identify": 2, "ripe": 3, "predict": 2, "type of": 2, "label": 1, "grad-cam": 2,
                 "gradcam": 2, "phan loai": 3, "nhan dien": 3, "qua gi": 3, "la gi": 2, "chin": 3,
                 "du doan": 2, "loai": 1},
    "measure": {"measur": 3, "size": 3, "diameter": 3, "how big": 3, "depth": 3, "disparity": 3,
                "stereo": 3, "distance": 2, "count": 2, "how many": 2, "kich thuoc": 3,
                "duong kinh": 3, "do sau": 3, "khoang cach": 2, "dem": 2, "bao nhieu": 2, "to nho": 2},
    "inspect": {"inspect": 3, "full report": 3, "everything": 2, "all": 1, "check this": 2,
                "kiem tra": 3, "bao cao": 2, "tat ca": 2},
}

HELP = ("I can: **calibrate** (camera K + undistort), **filter/denoise** (box | gaussian | median), "
        "**edges** (Sobel, Canny), **frequency** (FFT low/high-pass), **features** (Harris / SIFT / ORB "
        "matching with ratio test), **classify** (CNN + Grad-CAM), **measure** (object size / stereo depth) "
        "and **inspect** (classify + measure + edges).")


def normalise_text(s: str) -> str:
    """lower-case + remove Vietnamese accents so 'phân loại' == 'phan loai'."""
    s = s.lower().replace("đ", "d")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return " " + re.sub(r"\s+", " ", s) + " "


@dataclass
class Intent:
    name: str
    confidence: float
    params: dict
    scores: dict


def extract_params(text: str) -> dict:
    t = normalise_text(text)
    p = {}
    for m in ("median", "gaussian", "box"):
        if m in t:
            p["method"] = m
            break
    for m in ("sift", "orb"):
        if m in t:
            p["feat"] = m
    if "canny" in t:
        p["edge"] = "canny"
    if "high" in t or "cao" in t:
        p["pass"] = "high"
    if "add noise" in t or "them nhieu" in t or "salt" in t:
        p["add_noise"] = "salt_pepper" if ("salt" in t or "pepper" in t or "muoi" in t) else "gaussian"
    m = re.search(r"(?:ksize|kernel|size|kich thuoc loc)?\s*(\d+)\s*x\s*\d+", t) or \
        re.search(r"(?:median|gaussian|box|ksize|kernel)\s*(\d{1,2})\b", t)
    if m:
        k = int(m.group(1))
        p["ksize"] = k if k % 2 == 1 else k + 1
    m = re.search(r"ratio\s*(?:=|of)?\s*(0?\.\d+)", t)
    if m:
        p["ratio"] = float(m.group(1))
    m = re.search(r"sigma\s*(?:=)?\s*(\d+(?:\.\d+)?)", t)
    if m:
        p["sigma"] = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*mm\s*(?:/|per)\s*(?:px|pixel)", t)
    if m:
        p["mm_per_px"] = float(m.group(1))
    return p


def recognise(text: str, n_images: int = 1) -> Intent:
    """Score each intent by weighted keyword hits (+ small context priors)."""
    t = normalise_text(text)
    scores = {k: 0.0 for k in INTENTS}
    for intent, kws in INTENTS.items():
        for kw, w in kws.items():
            # long keywords are prefixes ("calibrat" -> calibrate/calibration),
            # short ones must be whole words ("pho" must not match "photo")
            pat = r"\b" + re.escape(kw) + (r"s?\b" if len(kw) < 5 else "")
            if re.search(pat, t):
                scores[intent] += w
    if n_images >= 2:                       # two images -> matching / stereo are more likely
        scores["features"] += 0.5
        scores["measure"] += 0.25
    total = sum(scores.values())
    best = max(scores, key=scores.get)
    if total == 0:
        return Intent("help", 0.0, {}, scores)
    return Intent(best, scores[best] / total, extract_params(text), scores)


# ---------------------------------------------------------------------------
# 2) tools (each returns a ToolResult)
# ---------------------------------------------------------------------------
@dataclass
class ToolResult:
    title: str
    text: str = ""
    table: list = field(default_factory=list)          # list of dicts
    images: list = field(default_factory=list)         # (caption, RGB/gray array)


def _rgb(b):
    return cv2.cvtColor(b, cv2.COLOR_BGR2RGB) if b.ndim == 3 else b


_CALIB = None
_CLF = None


def get_calibration():
    global _CALIB
    if _CALIB is None:
        _CALIB = C.default_calibration(os.path.join(ROOT, "data"))
    return _CALIB


def get_classifier():
    global _CLF
    if _CLF is None:
        from dl.model import FruitClassifier
        _CLF = FruitClassifier(os.path.join(ROOT, "models"))
    return _CLF


def tool_calibrate(images, p):
    """If >= 3 checkerboard photos are given -> calibrate them; else use the saved calibration."""
    note = ""
    calib = None
    if len(images) >= 3:
        tmp = []
        for i, im in enumerate(images):
            fn = os.path.join(ROOT, "outputs", f"_upload_calib_{i}.png")
            cv2.imwrite(fn, im); tmp.append(fn)
        try:
            calib = C.calibrate_from_images(tmp)
            note = f"Calibrated from your {calib.n_views} checkerboard photos."
        except RuntimeError as e:
            note = f"Could not calibrate from the uploads ({e}); using the saved calibration."
    if calib is None:
        calib = get_calibration()
        note = note or f"Using the saved calibration ({calib.n_views} views of a 9x6 board)."
    K, d = calib.K, calib.dist.ravel()
    table = [{"parameter": "fx", "value": round(float(K[0, 0]), 2)}, {"parameter": "fy", "value": round(float(K[1, 1]), 2)},
             {"parameter": "cx", "value": round(float(K[0, 2]), 2)}, {"parameter": "cy", "value": round(float(K[1, 2]), 2)},
             {"parameter": "k1, k2, p1, p2, k3", "value": ", ".join(f"{v:.4f}" for v in d[:5])},
             {"parameter": "RMS reprojection error (px)", "value": round(float(calib.rms), 3)}]
    imgs = []
    if images:
        im = images[0]
        if im.shape[:2] != (calib.image_size[1], calib.image_size[0]):
            note += " (Note: the image size differs from the calibrated camera, the result is only indicative.)"
        imgs = [("input + detected corners", _rgb(C.draw_corners(im))), ("undistorted", _rgb(C.undistort(im, calib)))]
    return ToolResult("Camera calibration", note, table, imgs)


def tool_filter(images, p):
    im = images[0]
    g = F.to_gray01(im)
    method, k = p.get("method", "median"), p.get("ksize", 5)
    src, table = g, []
    if p.get("add_noise"):
        src = F.add_gaussian_noise(g, 0.08) if p["add_noise"] == "gaussian" else F.add_salt_pepper(g, 0.06)
        for m, (e, ps) in F.compare_denoisers(g, src, k).items():
            table.append({"filter": f"{m} {k}x{k}", "MSE": round(float(e), 5), "PSNR (dB)": round(float(ps), 2)})
    out = F.denoise(src, method, k, p.get("sigma", 1.0))
    txt = f"Applied a {method} {k}x{k} filter."
    if table:
        best = min(table, key=lambda r: r["MSE"])
        txt += f" Added {p['add_noise']} noise first; best filter here: **{best['filter']}**."
    return ToolResult("Filtering / denoising", txt, table,
                      [("input" + (" + noise" if p.get("add_noise") else ""), src), (f"{method} {k}x{k}", out)])


def tool_edges(images, p):
    im = images[0]
    g = F.to_gray01(im)
    _, _, mag, ori = F.sobel_gradients(g, p.get("sigma", 1.0))
    e = F.canny(np.uint8(g * 255))
    return ToolResult("Gradients & edges",
                      f"Sobel on a Gaussian-smoothed image (sigma={p.get('sigma', 1.0)}) + Canny(50,150). "
                      f"Edge pixels: {100 * (e > 0).mean():.1f} %.",
                      [], [("|∇I| (Sobel)", mag / (mag.max() + 1e-9)), ("Canny edges", e)])


def tool_frequency(images, p):
    g = F.to_gray01(images[0])
    s = p.get("sigma", 25.0)
    hp = p.get("pass") == "high"
    out = F.freq_filter(g, F.gaussian_freq_mask(g.shape, s, highpass=hp))
    spec = F.log_spectrum(g)
    out_v = (out - out.min()) / (out.max() - out.min() + 1e-9)
    return ToolResult("Frequency-domain filtering",
                      f"Gaussian {'high' if hp else 'low'}-pass in the Fourier domain, sigma_f = {s}.", [],
                      [("log |F| spectrum", spec / spec.max()), (("high" if hp else "low") + "-pass result", out_v)])


def tool_features(images, p):
    meth = p.get("feat", "sift")
    if len(images) >= 2:
        st, vis = FT.match_two_views(images[0], images[1], meth, p.get("ratio", 0.75))
        same = st["ransac_inliers"] >= 15
        txt = (f"{meth.upper()}: {st['ratio_matches']} matches pass the ratio test "
               f"({p.get('ratio', 0.75)}), {st['ransac_inliers']} are geometrically consistent (RANSAC). "
               + ("→ **Same object/scene found.**" if same else "→ probably **not** the same object."))
        return ToolResult("Feature matching", txt, [st], [("inlier matches", _rgb(vis))])
    pts = FT.harris_corners(F.to_gray01(images[0]))
    kps, _ = FT.detect_and_describe(cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY), meth)
    vis2 = cv2.drawKeypoints(images[0], kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return ToolResult("Interest points", f"{len(pts)} Harris corners (my implementation), {len(kps)} {meth.upper()} "
                      "keypoints. Upload a second image to match them.", [],
                      [("Harris corners", _rgb(FT.draw_points(images[0], pts))), (f"{meth.upper()} keypoints", _rgb(vis2))])


def tool_classify(images, p):
    from dl.model import overlay_heatmap
    clf = get_classifier()
    top, _ = clf.predict(images[0])
    cam = clf.gradcam(images[0])
    name, conf = top[0]
    advice = ""
    if name.startswith("Tomato"):
        advice = " Ripeness: " + ("**not ripe yet** – keep on the plant." if "not Ripened" in name
                                  else "**ripe** – ready to harvest.")
    warn = "" if conf >= 0.6 else " ⚠️ Low confidence – the photo may be outside the training classes."
    return ToolResult("Classification (CNN)", f"Prediction: **{name}** ({100 * conf:.1f} %).{advice}{warn}",
                      [{"class": c, "probability": round(float(pp), 4)} for c, pp in top],
                      [("input", _rgb(images[0])), ("Grad-CAM (where the CNN looks)", _rgb(overlay_heatmap(images[0], cam)))])


def tool_measure(images, p):
    if len(images) >= 2:
        disp, sc = M.disparity_sgbm(images[0], images[1], num_disp=p.get("ndisp", 96))
        txt = (f"Stereo disparity (SGBM) on the image pair (resized x{sc:.2f}). Valid pixels: "
               f"{100 * np.isfinite(disp).mean():.0f} %. Bigger disparity = closer (Z = f·B / d).")
        return ToolResult("Depth from stereo", txt,
                          [{"disparity p5 (px)": round(float(np.nanpercentile(disp, 5)), 1),
                            "disparity p95 (px)": round(float(np.nanpercentile(disp, 95)), 1)}],
                          [("left", _rgb(cv2.resize(images[0], None, fx=sc, fy=sc))), ("disparity", _rgb(M.colorize(disp)))])
    rows, vis, mask = M.measure_objects(images[0], mm_per_px=p.get("mm_per_px"))
    txt = f"Found {len(rows)} object(s)."
    if not p.get("mm_per_px"):
        txt += " Sizes are in pixels; add e.g. '0.5 mm/px' to convert."
    return ToolResult("Object measurement", txt, rows, [("segmentation", mask), ("measured objects", _rgb(vis))])


def tool_inspect(images, p):
    parts = [tool_classify(images, p), tool_measure(images[:1], p), tool_edges(images, p)]
    return ToolResult("Full inspection", "  \n".join(f"*{r.title}:* {r.text}" for r in parts),
                      parts[0].table + parts[1].table, sum((r.images for r in parts), []))


TOOLS = {"calibrate": tool_calibrate, "filter": tool_filter, "edges": tool_edges, "frequency": tool_frequency,
         "features": tool_features, "classify": tool_classify, "measure": tool_measure, "inspect": tool_inspect}


def run(text: str, images: list, forced_intent: str | None = None):
    """Main entry used by the UI: recognise -> route -> execute."""
    intent = recognise(text or "", len(images))
    if forced_intent and forced_intent != "auto":
        intent = Intent(forced_intent, 1.0, extract_params(text or ""), intent.scores)
    if intent.name == "help":
        return intent, ToolResult("Help", "Sorry, I did not understand. " + HELP)
    if not images and intent.name != "calibrate":
        return intent, ToolResult("Need an image", "Please upload an image first.")
    return intent, TOOLS[intent.name](images, intent.params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--save", default=None, help="folder to save the result images")
    a = ap.parse_args()
    imgs = [cv2.imread(f) for f in a.image]
    intent, res = run(a.text, imgs)
    print(f"intent = {intent.name} (conf {intent.confidence:.2f}) params = {intent.params}")
    print(res.title, "-", res.text)
    for r in res.table:
        print("  ", r)
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        for i, (cap, im) in enumerate(res.images):
            im = np.asarray(im)
            if im.dtype != np.uint8:
                im = np.uint8(np.clip(im, 0, 1) * 255)
            cv2.imwrite(os.path.join(a.save, f"{intent.name}_{i}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR) if im.ndim == 3 else im)


if __name__ == "__main__":
    main()
