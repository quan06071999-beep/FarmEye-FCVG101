"""FarmEye – Streamlit UI for the visual inspection assistant (Task 3).

run:  streamlit run app/streamlit_app.py
"""
import os
import sys

import cv2
import numpy as np
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from app import assistant as A  # noqa: E402

st.set_page_config(page_title="FarmEye – visual inspection assistant", page_icon="🍅", layout="wide")


@st.cache_resource
def warm_up():
    """Load the calibration + CNN once (they are cached inside the assistant module)."""
    A.get_calibration()
    try:
        A.get_classifier()
        return True
    except Exception as e:      # model file missing -> other tools still work
        st.warning(f"CNN model not loaded: {e}")
        return False


def to_bgr(uploaded):
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def show_img(cap, im):
    im = np.asarray(im)
    if im.dtype != np.uint8:
        im = np.clip(im, 0, 1)
    st.image(im, caption=cap, use_container_width=True, clamp=True)


warm_up()
st.title("🍅 FarmEye – visual inspection assistant")
st.caption("Organic-farm MVP · camera calibration · classical CV pipeline · CNN fruit/vegetable classifier")

with st.sidebar:
    st.header("1. Images")
    files = st.file_uploader("Upload 1 image (or 2 for matching / stereo, ≥3 checkerboards to calibrate)",
                             type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)
    demo = st.selectbox("…or use a demo image set", [
        "(none)", "fruit: tomato (test set)", "fruit: banana (test set)", "checkerboard (left12.jpg)",
        "feature pair: box + scene", "stereo pair: aloe L/R", "still life: fruits.jpg"])
    st.header("2. Mode")
    mode = st.radio("Tool", ["auto (understand my sentence)", "calibrate", "filter", "edges", "frequency",
                             "features", "classify", "measure", "inspect"], index=0)
    st.markdown(A.HELP)

DEMOS = {
    "fruit: tomato (test set)": ["data/fruits/Test/Tomato 1/{}"],
    "fruit: banana (test set)": ["data/fruits/Test/Banana/{}"],
    "checkerboard (left12.jpg)": ["data/calib/left12.jpg"],
    "feature pair: box + scene": ["data/features/box.png", "data/features/box_in_scene.png"],
    "stereo pair: aloe L/R": ["data/stereo/aloeL.jpg", "data/stereo/aloeR.jpg"],
    "still life: fruits.jpg": ["data/samples/fruits.jpg"],
}

images = [to_bgr(f) for f in files] if files else []
if not images and demo != "(none)":
    for p in DEMOS[demo]:
        full = os.path.join(ROOT, p)
        if "{}" in full:                         # first file of that folder
            d = os.path.dirname(full)
            full = os.path.join(d, sorted(os.listdir(d))[0])
        images.append(cv2.imread(full))

if images:
    cols = st.columns(4)
    for i, im in enumerate(images[:4]):
        with cols[i]:
            st.image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), caption=f"input {i + 1}  {im.shape[1]}x{im.shape[0]}",
                     use_container_width=True)

text = st.chat_input("Ask me something, e.g. 'is this tomato ripe?', 'khử nhiễu bằng median 7', "
                     "'match these two photos with ORB'")
if "history" not in st.session_state:
    st.session_state.history = []

if text or (mode != "auto (understand my sentence)" and st.button(f"Run '{mode}'")):
    forced = None if mode.startswith("auto") else mode
    with st.spinner("working…"):
        intent, res = A.run(text or "", images, forced)
    st.session_state.history.insert(0, (text or f"[{mode}]", intent, res))

for q, intent, res in st.session_state.history[:3]:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.markdown(f"**Detected task:** `{intent.name}` · confidence {intent.confidence:.2f} · "
                    f"params `{intent.params}`")
        st.subheader(res.title)
        st.markdown(res.text)
        if res.table:
            st.dataframe(res.table, use_container_width=True)
        if res.images:
            cols = st.columns(3)
            for i, (cap, im) in enumerate(res.images):
                with cols[i % len(cols)]:
                    show_img(cap, im)
