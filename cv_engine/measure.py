"""Measurement tools: stereo disparity / depth and simple object size measuring."""
from __future__ import annotations

import cv2
import numpy as np


def disparity_sgbm(left_bgr, right_bgr, num_disp: int = 128, block: int = 5, max_width: int = 640):
    """Semi-global block matching on a rectified pair. Returns disparity (px) at the
    resized resolution and the scale used."""
    scale = min(1.0, max_width / left_bgr.shape[1])
    L = cv2.resize(left_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    R = cv2.resize(right_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gl, gr = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    num_disp = int(np.ceil(num_disp / 16) * 16)
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disp, blockSize=block,
                                 P1=8 * block * block, P2=32 * block * block,
                                 disp12MaxDiff=1, uniquenessRatio=10,
                                 speckleWindowSize=100, speckleRange=2,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(gl, gr).astype(np.float32) / 16.0     # fixed point -> pixels
    disp[disp <= 0] = np.nan                                   # invalid
    return disp, scale


def depth_from_disparity(disp_px, focal_px: float, baseline_mm: float):
    """Z = f * B / d   (rectified stereo)."""
    return focal_px * baseline_mm / disp_px


def colorize(arr, cmap=cv2.COLORMAP_JET):
    a = np.nan_to_num(arr, nan=np.nanmin(arr))
    a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(a, cmap)


def measure_objects(img_bgr, min_area: int = 300, mm_per_px: float | None = None):
    """Segment objects that differ from the background and measure them.

    Background colour = median of the image border (Lab space); the per-pixel
    colour distance to it is thresholded with Otsu. Works for the white
    background of Fruits-360 and for darker table/conveyor backgrounds.
    Reports area, equivalent diameter and bounding box of each object.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[:5].reshape(-1, 3), lab[-5:].reshape(-1, 3),
                             lab[:, :5].reshape(-1, 3), lab[:, -5:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    score = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = img_bgr.copy()
    rows = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        d = float(np.sqrt(4 * a / np.pi))
        row = {"id": len(rows) + 1, "area_px": int(a), "eq_diameter_px": round(d, 1),
               "bbox": [int(x), int(y), int(w), int(h)]}
        if mm_per_px:
            row["eq_diameter_mm"] = round(d * mm_per_px, 1)
        rows.append(row)
        cv2.drawContours(vis, [c], -1, (0, 255, 0), 2)
        cv2.putText(vis, f"#{row['id']} d={d:.0f}px", (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return rows, vis, mask
