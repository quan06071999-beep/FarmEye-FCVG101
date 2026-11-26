"""Interest points and matching.

- Harris corner detector implemented from scratch (structure tensor + NMS)
- SIFT / ORB with OpenCV
- my own brute-force matcher with Lowe's ratio test (compared to cv2.BFMatcher)
- RANSAC homography to count geometrically consistent matches (inliers)
"""
from __future__ import annotations

import cv2
import numpy as np

from .filtering import correlate2d, gaussian_kernel, SOBEL_X, SOBEL_Y


# --------------------------------------------------------------------------
# Harris from scratch
# --------------------------------------------------------------------------
def harris_response(gray01: np.ndarray, sigma_d: float = 1.0, sigma_i: float = 2.0, k: float = 0.05):
    """R = det(M) - k trace(M)^2 with M = G_sigma_i * [Ix^2 IxIy; IxIy Iy^2]."""
    kd = int(2 * round(3 * sigma_d) + 1)
    g = correlate2d(gray01, gaussian_kernel(kd, sigma_d))
    Ix = correlate2d(g, SOBEL_X) / 8.0
    Iy = correlate2d(g, SOBEL_Y) / 8.0
    ki = int(2 * round(3 * sigma_i) + 1)
    W = gaussian_kernel(ki, sigma_i)
    Sxx = correlate2d(Ix * Ix, W)
    Syy = correlate2d(Iy * Iy, W)
    Sxy = correlate2d(Ix * Iy, W)
    return (Sxx * Syy - Sxy ** 2) - k * (Sxx + Syy) ** 2


def harris_corners(gray01: np.ndarray, max_corners: int = 300, rel_thresh: float = 0.01,
                   nms_size: int = 9, **kw) -> np.ndarray:
    """Return (N,2) array of (x, y) corner locations after non-maximum suppression."""
    R = harris_response(gray01, **kw)
    dil = cv2.dilate(R, np.ones((nms_size, nms_size), np.uint8))      # local maxima
    mask = (R == dil) & (R > rel_thresh * R.max())
    ys, xs = np.nonzero(mask)
    order = np.argsort(-R[ys, xs])[:max_corners]
    return np.stack([xs[order], ys[order]], axis=1)


# --------------------------------------------------------------------------
# descriptors + matching
# --------------------------------------------------------------------------
def detect_and_describe(gray_u8: np.ndarray, method: str = "sift", n_features: int = 2000):
    if method == "sift":
        det = cv2.SIFT_create(nfeatures=n_features)
    elif method == "orb":
        det = cv2.ORB_create(nfeatures=n_features)
    else:
        raise ValueError(method)
    kps, desc = det.detectAndCompute(gray_u8, None)
    return kps, desc


def ratio_test_match(d1: np.ndarray, d2: np.ndarray, ratio: float = 0.75, binary: bool = False):
    """My brute-force 2-NN matcher + Lowe ratio test.

    Returns list of (i, j, dist) with i in d1, j in d2.
    """
    if d1 is None or d2 is None or len(d1) < 1 or len(d2) < 2:
        return []
    if binary:   # ORB -> Hamming distance on unpacked bits
        b1 = np.unpackbits(d1, axis=1).astype(np.int16)
        b2 = np.unpackbits(d2, axis=1).astype(np.int16)
        D = b1 @ (1 - b2).T + (1 - b1) @ b2.T
    else:        # SIFT -> Euclidean distance, ||a-b||^2 = a^2 + b^2 - 2ab
        a = d1.astype(np.float64); b = d2.astype(np.float64)
        D = np.sqrt(np.maximum((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T, 0))
    nn = np.argsort(D, axis=1)[:, :2]
    best = D[np.arange(len(D)), nn[:, 0]]
    second = D[np.arange(len(D)), nn[:, 1]]
    keep = best < ratio * second
    return [(int(i), int(nn[i, 0]), float(best[i])) for i in np.nonzero(keep)[0]]


def opencv_ratio_match(d1, d2, ratio=0.75, binary=False):
    """Reference: cv2.BFMatcher.knnMatch + ratio test."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING if binary else cv2.NORM_L2)
    out = []
    for pair in bf.knnMatch(d1, d2, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            out.append((pair[0].queryIdx, pair[0].trainIdx, pair[0].distance))
    return out


def ransac_homography(kp1, kp2, matches, thresh: float = 5.0):
    """Fit H with RANSAC; return H and boolean inlier mask (None if < 4 matches)."""
    if len(matches) < 4:
        return None, np.zeros(len(matches), bool)
    p1 = np.float32([kp1[i].pt for i, _, _ in matches])
    p2 = np.float32([kp2[j].pt for _, j, _ in matches])
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, thresh)
    return H, (mask.ravel().astype(bool) if mask is not None else np.zeros(len(matches), bool))


def match_two_views(img1_bgr, img2_bgr, method="sift", ratio=0.75):
    """Full pipeline: detect, describe, ratio-test match, RANSAC. Returns dict + vis image."""
    g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY) if img1_bgr.ndim == 3 else img1_bgr
    g2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY) if img2_bgr.ndim == 3 else img2_bgr
    kp1, d1 = detect_and_describe(g1, method)
    kp2, d2 = detect_and_describe(g2, method)
    m = ratio_test_match(d1, d2, ratio, binary=(method == "orb"))
    H, inl = ransac_homography(kp1, kp2, m)
    dm = [cv2.DMatch(i, j, d) for (i, j, d) in m]
    vis = cv2.drawMatches(img1_bgr, kp1, img2_bgr, kp2, [d for d, ok in zip(dm, inl) if ok], None,
                          matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if H is not None:                                    # outline the object in the scene
        h, w = g1.shape
        box = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), H)
        box[:, :, 0] += w
        cv2.polylines(vis, [np.int32(box)], True, (255, 0, 0), 3)
    stats = {"method": method, "kp1": len(kp1), "kp2": len(kp2), "ratio_matches": len(m),
             "ransac_inliers": int(inl.sum()),
             "inlier_ratio": round(float(inl.mean()), 3) if len(m) else 0.0}
    return stats, vis


def draw_points(img_bgr, pts_xy, color=(0, 0, 255), r=4):
    vis = img_bgr.copy()
    for x, y in pts_xy:
        cv2.circle(vis, (int(x), int(y)), r, color, 1, cv2.LINE_AA)
    return vis
