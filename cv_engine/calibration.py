"""Image formation & camera calibration (Requirement a).

- pinhole projection  x = K [R|t] X
- checkerboard calibration with cv2.calibrateCamera (+ sub-pixel corners)
- undistortion
- my own DLT / homography (from scratch, numpy SVD) to compare with OpenCV
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Pinhole model
# --------------------------------------------------------------------------
def project_points(Xw: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray,
                   dist: np.ndarray | None = None) -> np.ndarray:
    """Project (N,3) world points to pixels with x ~ K (R Xw + t).

    If `dist` (k1, k2, p1, p2, k3) is given the Brown-Conrady lens distortion is
    applied on the normalised coordinates before K, like cv2.projectPoints.
    """
    Xc = (R @ np.asarray(Xw, np.float64).T + np.asarray(t, np.float64).reshape(3, 1)).T
    x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]          # normalised image plane
    if dist is not None:
        k1, k2, p1, p2, k3 = np.ravel(dist)[:5]
        r2 = x ** 2 + y ** 2
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
        yd = y * radial + p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y
        x, y = xd, yd
    u = K[0, 0] * x + K[0, 1] * y + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return np.stack([u, v], axis=1)


# --------------------------------------------------------------------------
# DLT homography from scratch (planar target, Z = 0)
# --------------------------------------------------------------------------
def _normalise(pts: np.ndarray):
    """Hartley normalisation: centroid -> 0, mean distance -> sqrt(2)."""
    c = pts.mean(axis=0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(pts - c, axis=1))
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    ph = np.c_[pts, np.ones(len(pts))] @ T.T
    return ph, T


def dlt_homography(src_xy: np.ndarray, dst_uv: np.ndarray) -> np.ndarray:
    """Estimate H (3x3) with dst ~ H src using the normalised DLT + SVD."""
    a, Ta = _normalise(np.asarray(src_xy, np.float64))
    b, Tb = _normalise(np.asarray(dst_uv, np.float64))
    A = []
    for (x, y, _), (u, v, _) in zip(a, b):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tb) @ Hn @ Ta                 # undo normalisation
    return H / H[2, 2]


def intrinsics_from_homographies(Hs: list[np.ndarray]) -> np.ndarray:
    """Zhang's closed-form K from >= 3 plane homographies (no distortion)."""
    def v(H, i, j):
        h = H.T
        return np.array([h[i, 0] * h[j, 0],
                         h[i, 0] * h[j, 1] + h[i, 1] * h[j, 0],
                         h[i, 1] * h[j, 1],
                         h[i, 2] * h[j, 0] + h[i, 0] * h[j, 2],
                         h[i, 2] * h[j, 1] + h[i, 1] * h[j, 2],
                         h[i, 2] * h[j, 2]])
    V = []
    for H in Hs:
        V.append(v(H, 0, 1))
        V.append(v(H, 0, 0) - v(H, 1, 1))
    _, _, Vt = np.linalg.svd(np.asarray(V))
    b = Vt[-1]
    B11, B12, B22, B13, B23, B33 = b
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 ** 2)
    lam = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(lam / B11)
    beta = np.sqrt(lam * B11 / (B11 * B22 - B12 ** 2))
    gamma = -B12 * alpha ** 2 * beta / lam
    u0 = gamma * v0 / beta - B13 * alpha ** 2 / lam
    return np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])


# --------------------------------------------------------------------------
# Checkerboard calibration with OpenCV
# --------------------------------------------------------------------------
@dataclass
class CalibrationResult:
    K: np.ndarray
    dist: np.ndarray
    rms: float
    image_size: tuple
    n_views: int
    per_view_error: list = field(default_factory=list)
    K_zhang_scratch: np.ndarray | None = None
    used_files: list = field(default_factory=list)
    rejected: dict = field(default_factory=dict)
    rms_before_rejection: float | None = None

    def to_json(self, path: str):
        d = {"K": self.K.tolist(), "dist": self.dist.ravel().tolist(), "rms_px": self.rms,
             "image_size": list(self.image_size), "n_views": self.n_views,
             "per_view_error_px": self.per_view_error, "used_files": self.used_files,
             "rejected_views": self.rejected, "rms_before_rejection": self.rms_before_rejection,
             "K_zhang_scratch": None if self.K_zhang_scratch is None else self.K_zhang_scratch.tolist()}
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @staticmethod
    def from_json(path: str) -> "CalibrationResult":
        with open(path) as f:
            d = json.load(f)
        return CalibrationResult(np.array(d["K"]), np.array(d["dist"]), d["rms_px"],
                                 tuple(d["image_size"]), d["n_views"], d["per_view_error_px"],
                                 None if d.get("K_zhang_scratch") is None else np.array(d["K_zhang_scratch"]),
                                 d.get("used_files", []))


def find_corners(gray: np.ndarray, pattern=(9, 6)):
    """Detect inner checkerboard corners with sub-pixel refinement."""
    ok, corners = cv2.findChessboardCorners(gray, pattern,
                                            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit).reshape(-1, 1, 2)


def calibrate_from_images(paths: list[str], pattern=(9, 6), square_mm=25.0,
                          reject_factor: float | None = 3.0) -> CalibrationResult:
    """Calibrate a camera from checkerboard photos (cv2.calibrateCamera).

    reject_factor: views whose RMS error is > reject_factor x median are removed
    and the calibration is run again (bad corner detection / motion blur).
    """
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_mm
    obj_pts, img_pts, used, size = [], [], [], None
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        c = find_corners(gray, pattern)
        if c is not None:
            obj_pts.append(objp)
            img_pts.append(c)
            used.append(os.path.basename(p))
    if len(obj_pts) < 3:
        raise RuntimeError("need at least 3 views with a detected checkerboard")

    res = _calibrate_lists(obj_pts, img_pts, used, size)
    if reject_factor is not None and len(obj_pts) > 4:
        med = float(np.median(res.per_view_error))
        bad = [i for i, e in enumerate(res.per_view_error) if e > reject_factor * med]
        if bad and len(obj_pts) - len(bad) >= 3:
            keep = [i for i in range(len(obj_pts)) if i not in bad]
            res2 = _calibrate_lists([obj_pts[i] for i in keep], [img_pts[i] for i in keep],
                                    [used[i] for i in keep], size)
            res2.rejected = {used[i]: round(res.per_view_error[i], 3) for i in bad}
            res2.rms_before_rejection = res.rms
            return res2
    return res


def _calibrate_lists(obj_pts, img_pts, used, size) -> CalibrationResult:
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    per_view = []
    for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        e = proj.reshape(-1, 2) - i.reshape(-1, 2)          # OpenCV 4: (N,1,2), OpenCV 5: (N,2)
        per_view.append(float(np.sqrt(np.mean(np.sum(e ** 2, axis=1)))))
    return _finish(obj_pts, img_pts, used, size, rms, K, dist, per_view)


def _finish(obj_pts, img_pts, used, size, rms, K, dist, per_view) -> CalibrationResult:
    # from-scratch check: homography per view (my DLT) -> Zhang closed-form K
    Hs = [dlt_homography(o[:, :2], i.reshape(-1, 2)) for o, i in zip(obj_pts, img_pts)]
    K_scratch = intrinsics_from_homographies(Hs)

    return CalibrationResult(K, dist, float(rms), size, len(obj_pts), per_view, K_scratch, used)


def undistort(img: np.ndarray, calib: CalibrationResult, alpha: float = 0.0) -> np.ndarray:
    """Remove lens distortion. alpha=0 crops to valid pixels, 1 keeps all."""
    h, w = img.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(calib.K, calib.dist, (w, h), alpha, (w, h))
    out = cv2.undistort(img, calib.K, calib.dist, None, newK)
    return out


def draw_corners(img: np.ndarray, pattern=(9, 6)) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    c = find_corners(gray, pattern)
    if c is not None:
        cv2.drawChessboardCorners(vis, pattern, c, True)
    return vis


def default_calibration(data_dir: str) -> CalibrationResult:
    """Load cached calibration (json) or run it on data/calib/*.jpg."""
    cache = os.path.join(data_dir, "..", "models", "calibration.json")
    if os.path.exists(cache):
        return CalibrationResult.from_json(cache)
    res = calibrate_from_images(sorted(glob.glob(os.path.join(data_dir, "calib", "*.jpg"))))
    res.to_json(cache)
    return res
