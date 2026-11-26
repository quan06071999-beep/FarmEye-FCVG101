"""Evaluate the intent recogniser on a small labelled set of user requests.

usage:  python tests/test_intents.py      (also works with pytest)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.assistant import recognise, extract_params  # noqa: E402

CASES = [
    # ---- English
    ("calibrate my camera with these checkerboard photos", 3, "calibrate"),
    ("undistort this image", 1, "calibrate"),
    ("what is the intrinsic matrix K of the camera?", 1, "calibrate"),
    ("remove the noise from this photo", 1, "filter"),
    ("denoise with median 7", 1, "filter"),
    ("smooth it with a gaussian filter, sigma 2", 1, "filter"),
    ("apply a 5x5 box filter", 1, "filter"),
    ("add salt and pepper noise then clean it", 1, "filter"),
    ("show me the edges", 1, "edges"),
    ("run canny edge detection", 1, "edges"),
    ("compute the sobel gradient magnitude", 1, "edges"),
    ("show the fourier spectrum", 1, "frequency"),
    ("high-pass filter in the frequency domain", 1, "frequency"),
    ("do an FFT low-pass", 1, "frequency"),
    ("detect harris corners", 1, "features"),
    ("find SIFT keypoints", 1, "features"),
    ("match these two photos with ORB, ratio 0.8", 2, "features"),
    ("is this the same box as in the reference photo?", 2, "features"),
    ("what fruit is this?", 1, "classify"),
    ("classify this image", 1, "classify"),
    ("is this tomato ripe?", 1, "classify"),
    ("identify the vegetable and show grad-cam", 1, "classify"),
    ("measure the size of the fruit", 1, "measure"),
    ("how big is this lemon in mm? 0.5 mm/px", 1, "measure"),
    ("estimate depth from this stereo pair", 2, "measure"),
    ("how many objects are on the table?", 1, "measure"),
    ("inspect this crate and give me a full report", 1, "inspect"),
    ("check this photo please", 1, "inspect"),
    # ---- Vietnamese (with and without accents)
    ("hiệu chỉnh camera bằng ảnh bàn cờ", 3, "calibrate"),
    ("sửa méo ảnh này", 1, "calibrate"),
    ("khử nhiễu ảnh bằng median 7", 1, "filter"),
    ("làm mờ ảnh bằng bộ lọc gaussian", 1, "filter"),
    ("tìm biên của ảnh", 1, "edges"),
    ("tính đạo hàm sobel", 1, "edges"),
    ("phân tích phổ tần số", 1, "frequency"),
    ("so khớp đặc trưng giữa hai ảnh", 2, "features"),
    ("phân loại quả này", 1, "classify"),
    ("đây là quả gì?", 1, "classify"),
    ("quả cà chua này chín chưa?", 1, "classify"),
    ("đo kích thước quả", 1, "measure"),
    ("tính độ sâu từ cặp ảnh stereo", 2, "measure"),
    ("kiểm tra toàn bộ ảnh và báo cáo", 1, "inspect"),
    ("phan loai qua nay", 1, "classify"),
    ("khu nhieu anh", 1, "filter"),
]


def evaluate(verbose=True):
    ok = 0
    wrong = []
    for text, n, label in CASES:
        got = recognise(text, n).name
        ok += got == label
        if got != label:
            wrong.append((text, label, got))
    acc = ok / len(CASES)
    if verbose:
        print(f"intent accuracy: {ok}/{len(CASES)} = {acc:.3f}")
        for w in wrong:
            print("  WRONG:", w)
    return acc, wrong


def test_intent_accuracy():
    acc, _ = evaluate(False)
    assert acc >= 0.9


def test_params():
    p = extract_params("denoise with median 7")
    assert p["method"] == "median" and p["ksize"] == 7
    p = extract_params("match with ORB, ratio 0.8")
    assert p["feat"] == "orb" and abs(p["ratio"] - 0.8) < 1e-9
    assert extract_params("apply a 5x5 box filter")["ksize"] == 5
    assert extract_params("how big? 0.5 mm/px")["mm_per_px"] == 0.5


if __name__ == "__main__":
    evaluate()
    test_params()
    print("param extraction OK")
