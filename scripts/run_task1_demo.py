"""Run the whole Task 1 pipeline on the provided data and save figures + numbers.

usage:  python scripts/run_task1_demo.py
outputs: outputs/figures/t1_*.png, outputs/task1_results.json, models/calibration.json
"""
import glob
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cv_engine import calibration as C, filtering as F, features as FT, measure as M  # noqa: E402

FIG = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
res = {}


def rgb(b):
    return cv2.cvtColor(b, cv2.COLOR_BGR2RGB)


def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, name), dpi=95); plt.close()


# ---------------------------------------------------------------- 1.1 calibration
paths = sorted(glob.glob(os.path.join(ROOT, "data", "calib", "*.jpg")))
calib = C.calibrate_from_images(paths, pattern=(9, 6), square_mm=25.0)
calib.to_json(os.path.join(ROOT, "models", "calibration.json"))
res["calibration"] = {"n_images": len(paths), "n_views_used": calib.n_views, "rms_px": calib.rms,
                      "K": np.round(calib.K, 2).tolist(), "dist": np.round(calib.dist.ravel(), 4).tolist(),
                      "K_zhang_from_scratch": np.round(calib.K_zhang_scratch, 2).tolist(),
                      "per_view_rms_px": np.round(calib.per_view_error, 3).tolist(),
                      "rejected_views": calib.rejected, "rms_before_rejection": calib.rms_before_rejection,
                      "used_files": calib.used_files}

img = cv2.imread(paths[11])
und = C.undistort(img, calib, alpha=0.0)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
ax[0].imshow(rgb(C.draw_corners(cv2.imread(paths[0])))); ax[0].set_title("detected 9x6 corners (left01)")
ax[1].imshow(rgb(img)); ax[1].set_title("original (" + os.path.basename(paths[11]) + ")")
ax[2].imshow(rgb(und)); ax[2].set_title("undistorted")
for a in ax:
    a.axis("off")
save("t1_calibration.png")

# pinhole check: re-project the board of view 0 with my own projection function
objp = np.zeros((54, 3)); objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 25.0
gray0 = cv2.cvtColor(cv2.imread(paths[0]), cv2.COLOR_BGR2GRAY)
corners = C.find_corners(gray0)
ok, rvec, tvec = cv2.solvePnP(objp, corners, calib.K, calib.dist)
mine = C.project_points(objp, calib.K, cv2.Rodrigues(rvec)[0], tvec, calib.dist)
ref, _ = cv2.projectPoints(objp, rvec, tvec, calib.K, calib.dist)
res["pinhole_check"] = {"max_diff_vs_cv2_projectPoints_px": float(np.abs(mine - ref.reshape(-1, 2)).max()),
                        "reprojection_rms_px": float(np.sqrt(np.mean(np.sum((mine - corners.reshape(-1, 2)) ** 2, 1)))),
                        "board_distance_mm": float(np.linalg.norm(tvec))}

# ---------------------------------------------------------------- 1.2 filtering
fruit = cv2.imread(os.path.join(ROOT, "data", "samples", "fruits.jpg"))
g = F.to_gray01(fruit)
gn = F.add_gaussian_noise(g, 0.08, seed=1)
sp = F.add_salt_pepper(g, 0.06, seed=2)
table = {"gaussian_noise": F.compare_denoisers(g, gn), "salt_pepper": F.compare_denoisers(g, sp)}
res["denoising_mse_psnr"] = {k: {m: [round(v[0], 5), round(v[1], 2)] for m, v in d.items()} for k, d in table.items()}
res["noisy_psnr"] = {"gaussian_noise": round(F.psnr(gn, g), 2), "salt_pepper": round(F.psnr(sp, g), 2)}
# my correlate2d vs OpenCV
k = F.gaussian_kernel(5, 1.0)
res["correlate2d_vs_filter2D_maxdiff"] = float(np.abs(
    F.correlate2d(g, k) - cv2.filter2D(g, cv2.CV_64F, k, borderType=cv2.BORDER_REFLECT_101)).max())

fig, ax = plt.subplots(2, 4, figsize=(15, 7))
for r, (name, noisy) in enumerate((("Gaussian noise", gn), ("salt & pepper", sp))):
    ax[r, 0].imshow(noisy, cmap="gray"); ax[r, 0].set_title(f"{name} (PSNR {F.psnr(noisy, g):.1f} dB)")
    for c, m in enumerate(("box", "gaussian", "median")):
        d = F.denoise(noisy, m, 5)
        ax[r, c + 1].imshow(d, cmap="gray"); ax[r, c + 1].set_title(f"{m} 5x5 (PSNR {F.psnr(d, g):.1f} dB)")
for a in ax.ravel():
    a.axis("off")
save("t1_denoising.png")

gx, gy, mag, ori = F.sobel_gradients(g, pre_sigma=1.0)
edges = F.canny(np.uint8(g * 255), 50, 150)
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
ax[0].imshow(gx, cmap="RdBu_r"); ax[0].set_title("Ix (my Sobel)")
ax[1].imshow(mag, cmap="magma"); ax[1].set_title("|grad I|")
ax[2].imshow(ori, cmap="hsv"); ax[2].set_title("orientation")
ax[3].imshow(edges, cmap="gray"); ax[3].set_title("Canny (50, 150)")
for a in ax:
    a.axis("off")
save("t1_gradients.png")
res["edges"] = {"canny_edge_fraction": round(float((edges > 0).mean()), 4)}

spec = F.log_spectrum(g)
lp = F.freq_filter(g, F.gaussian_freq_mask(g.shape, 25))
hp = F.freq_filter(g, F.gaussian_freq_mask(g.shape, 25, highpass=True))
ideal = F.freq_filter(g, F.ideal_freq_mask(g.shape, 25))
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
ax[0].imshow(spec, cmap="inferno"); ax[0].set_title("log |F| (shifted)")
ax[1].imshow(lp, cmap="gray"); ax[1].set_title("Gaussian low-pass (sigma_f=25)")
ax[2].imshow(ideal, cmap="gray"); ax[2].set_title("ideal low-pass r=25 (ringing)")
ax[3].imshow(hp, cmap="gray"); ax[3].set_title("Gaussian high-pass")
for a in ax:
    a.axis("off")
save("t1_frequency.png")
res["frequency"] = {"lowpass_mse": round(F.mse(lp, g), 5), "ideal_lowpass_mse": round(F.mse(ideal, g), 5)}

# ---------------------------------------------------------------- 1.3 features
box = cv2.imread(os.path.join(ROOT, "data", "features", "box.png"))
scene = cv2.imread(os.path.join(ROOT, "data", "features", "box_in_scene.png"))
hc = FT.harris_corners(F.to_gray01(scene), max_corners=300)
res["harris_corners_scene"] = int(len(hc))
fstats = {}
for m in ("sift", "orb"):
    st, vis = FT.match_two_views(box, scene, m, ratio=0.75)
    fstats[m] = st
    plt.figure(figsize=(12, 5.2)); plt.imshow(rgb(vis)); plt.axis("off")
    plt.title(f"{m.upper()}: {st['ratio_matches']} ratio-test matches, {st['ransac_inliers']} RANSAC inliers")
    save(f"t1_matches_{m}.png")
# my matcher vs OpenCV BFMatcher (same descriptors)
gb, gs = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY), cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
k1, d1 = FT.detect_and_describe(gb, "sift"); k2, d2 = FT.detect_and_describe(gs, "sift")
mine_m = {(i, j) for i, j, _ in FT.ratio_test_match(d1, d2)}
cv_m = {(i, j) for i, j, _ in FT.opencv_ratio_match(d1, d2)}
fstats["my_matcher_vs_bfmatcher"] = {"mine": len(mine_m), "opencv": len(cv_m), "same_pairs": len(mine_m & cv_m)}
# effect of the ratio threshold
fstats["ratio_sweep_sift"] = {}
for r in (0.6, 0.7, 0.75, 0.8, 0.9):
    m = FT.ratio_test_match(d1, d2, r)
    _, inl = FT.ransac_homography(k1, k2, m)
    fstats["ratio_sweep_sift"][str(r)] = [len(m), int(inl.sum())]
res["features"] = fstats

plt.figure(figsize=(7, 5.5)); plt.imshow(rgb(FT.draw_points(scene, hc, (0, 0, 255), 4)))
plt.title(f"my Harris corners ({len(hc)})"); plt.axis("off"); save("t1_harris.png")

# ---------------------------------------------------------------- measure (stereo)
L = cv2.imread(os.path.join(ROOT, "data", "stereo", "aloeL.jpg"))
R = cv2.imread(os.path.join(ROOT, "data", "stereo", "aloeR.jpg"))
disp, sc = M.disparity_sgbm(L, R, num_disp=96)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].imshow(rgb(cv2.resize(L, None, fx=sc, fy=sc))); ax[0].set_title("left image (aloe)")
im = ax[1].imshow(disp, cmap="jet"); ax[1].set_title("SGBM disparity (px), near = red")
plt.colorbar(im, ax=ax[1], fraction=0.04)
for a in ax:
    a.axis("off")
save("t1_disparity.png")
res["stereo"] = {"scale": sc, "valid_fraction": round(float(np.isfinite(disp).mean()), 3),
                 "disp_range_px": [float(np.nanpercentile(disp, 2)), float(np.nanpercentile(disp, 98))]}

with open(os.path.join(ROOT, "outputs", "task1_results.json"), "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=1))
