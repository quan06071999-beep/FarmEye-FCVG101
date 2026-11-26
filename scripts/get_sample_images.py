"""Download the small OpenCV sample images used by Task 1 (checkerboards, box, aloe stereo pair).

usage:  python scripts/get_sample_images.py
They come from https://github.com/opencv/opencv/tree/4.x/samples/data (Apache 2.0).
"""
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"
FILES = {
    "calib": [f"left{i:02d}.jpg" for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14)],
    "features": ["box.png", "box_in_scene.png"],
    "stereo": ["aloeL.jpg", "aloeR.jpg"],
    "samples": ["fruits.jpg"],
}

if __name__ == "__main__":
    for folder, names in FILES.items():
        d = os.path.join(ROOT, "data", folder)
        os.makedirs(d, exist_ok=True)
        for n in names:
            dst = os.path.join(d, n)
            if not os.path.exists(dst):
                print("download", n)
                urllib.request.urlretrieve(BASE + n, dst)
    print("done ->", os.path.join(ROOT, "data"))
