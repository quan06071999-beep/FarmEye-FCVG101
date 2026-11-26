"""Linear / non-linear filtering, gradients and frequency-domain filtering.

From-scratch parts: correlate2d / convolve2d (vectorised with sliding windows),
gaussian_kernel, sobel gradients, gaussian frequency filters.
OpenCV parts are used as reference and for speed (box, median, Canny).
"""
from __future__ import annotations

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# --------------------------------------------------------------------------
# correlation / convolution from scratch
# --------------------------------------------------------------------------
def correlate2d(img: np.ndarray, kernel: np.ndarray, pad_mode: str = "reflect") -> np.ndarray:
    """2-D cross-correlation, output has the same size as `img`.

    Uses sliding_window_view + einsum instead of two python loops (same maths,
    ~100x faster). pad_mode: 'constant' (zero), 'reflect', 'edge'.
    """
    img = np.asarray(img, np.float64)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    if img.ndim == 3:                                   # colour: filter each channel
        return np.dstack([correlate2d(img[..., c], kernel, pad_mode) for c in range(img.shape[2])])
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode=pad_mode)
    windows = sliding_window_view(padded, (kh, kw))     # (H, W, kh, kw)
    return np.einsum("ijkl,kl->ij", windows, kernel.astype(np.float64))


def convolve2d(img: np.ndarray, kernel: np.ndarray, pad_mode: str = "reflect") -> np.ndarray:
    """Convolution = correlation with the kernel flipped in both directions."""
    return correlate2d(img, np.flip(kernel), pad_mode)


def box_kernel(size: int) -> np.ndarray:
    return np.ones((size, size)) / float(size * size)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    x = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()


# --------------------------------------------------------------------------
# noise models + metrics
# --------------------------------------------------------------------------
def add_gaussian_noise(img01: np.ndarray, sigma: float = 0.08, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(img01 + rng.normal(0, sigma, img01.shape), 0, 1)


def add_salt_pepper(img01: np.ndarray, amount: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img01.copy()
    mask = rng.random(img01.shape[:2])
    out[mask < amount / 2] = 0.0
    out[mask > 1 - amount / 2] = 1.0
    return out


def mse(a, b) -> float:
    return float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))


def psnr(a, b) -> float:
    m = mse(a, b)
    return float("inf") if m == 0 else 10 * np.log10(1.0 / m)


def denoise(img01: np.ndarray, method: str = "median", ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """img01 float in [0,1]. method: box | gaussian | median."""
    if method == "box":
        return correlate2d(img01, box_kernel(ksize))
    if method == "gaussian":
        return correlate2d(img01, gaussian_kernel(ksize, sigma))
    if method == "median":
        u8 = np.uint8(np.clip(img01, 0, 1) * 255)
        return cv2.medianBlur(u8, ksize).astype(np.float64) / 255.0
    raise ValueError(method)


def compare_denoisers(clean01: np.ndarray, noisy01: np.ndarray, ksize: int = 5) -> dict:
    """Return {method: (MSE, PSNR)} for box / gaussian / median."""
    out = {}
    for m in ("box", "gaussian", "median"):
        d = denoise(noisy01, m, ksize)
        out[m] = (mse(d, clean01), psnr(d, clean01))
    return out


# --------------------------------------------------------------------------
# gradients and edges
# --------------------------------------------------------------------------
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)
SOBEL_Y = SOBEL_X.T


def sobel_gradients(gray01: np.ndarray, pre_sigma: float = 1.0):
    """Smooth (Gaussian) then Sobel. Returns gx, gy, magnitude, orientation."""
    g = gray01.astype(np.float64)
    if pre_sigma > 0:
        k = int(2 * round(3 * pre_sigma) + 1)
        g = correlate2d(g, gaussian_kernel(k, pre_sigma))
    gx = correlate2d(g, SOBEL_X)
    gy = correlate2d(g, SOBEL_Y)
    return gx, gy, np.hypot(gx, gy), np.arctan2(gy, gx)


def canny(gray_u8: np.ndarray, low: int = 50, high: int = 150, blur_sigma: float = 1.2) -> np.ndarray:
    g = cv2.GaussianBlur(gray_u8, (0, 0), blur_sigma) if blur_sigma > 0 else gray_u8
    return cv2.Canny(g, low, high)


# --------------------------------------------------------------------------
# frequency domain
# --------------------------------------------------------------------------
def log_spectrum(gray01: np.ndarray) -> np.ndarray:
    F = np.fft.fftshift(np.fft.fft2(gray01))
    return np.log1p(np.abs(F))


def gaussian_freq_mask(shape, sigma_f: float, highpass: bool = False) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = (yy - h // 2) ** 2 + (xx - w // 2) ** 2
    m = np.exp(-d2 / (2 * sigma_f ** 2))
    return 1 - m if highpass else m


def ideal_freq_mask(shape, cutoff: float, highpass: bool = False) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    m = (np.hypot(yy - h // 2, xx - w // 2) <= cutoff).astype(np.float64)
    return 1 - m if highpass else m


def freq_filter(gray01: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Multiply the centred spectrum by `mask` and go back (convolution theorem)."""
    F = np.fft.fftshift(np.fft.fft2(gray01))
    return np.real(np.fft.ifft2(np.fft.ifftshift(F * mask)))


def to_gray01(img_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    return g.astype(np.float64) / 255.0
