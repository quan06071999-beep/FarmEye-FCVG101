# FarmEye – visual inspection assistant for an organic farm (FCVG101 coursework)

FarmEye is a small MVP computer-vision system for a fruit/vegetable sorting table on an organic farm.
It has three parts:

| Part | Folder | What it does |
|---|---|---|
| Task 1 – core image analysis engine | `cv_engine/` | pinhole model, checkerboard calibration (+ my own DLT/Zhang), undistortion, box/Gaussian/median filtering (my own correlation/convolution), Sobel/Canny, FFT filtering, Harris (from scratch), SIFT/ORB + ratio test (my own matcher) + RANSAC, stereo disparity, object size measurement |
| Task 2 – data pipeline + deep learning | `dl/` | Fruits-360 subset (8 classes) → ingestion, cleaning, stratified split, augmentation → VGG-style CNN (TensorFlow/Keras) → metrics, confusion matrix, robustness test, Grad-CAM → saved model `models/fruit_cnn.keras` |
| Task 3 – assistant (tool use) | `app/` | intent recognition from an English **or Vietnamese** sentence → calls the right tool → Streamlit UI shows images, tables and the CNN prediction |

```
fruit_inspector/
├── cv_engine/        calibration.py  filtering.py  features.py  measure.py
├── dl/               data_pipeline.py  model.py  train.py
├── app/              assistant.py (router + tools, also a CLI)  streamlit_app.py (GUI)
├── scripts/          run_task1_demo.py  get_fruits_data.py  get_sample_images.py  build_report.py
├── tests/            test_intents.py
├── data/             created by the download scripts: calib/ features/ stereo/ samples/ fruits/
├── models/           labels.json  calibration.json  (+ fruit_cnn.keras from the Drive link)
└── outputs/          metrics.json  task1_results.json  pipeline_report.json  train_*.json  cli_examples.txt
```

## 1. Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data and model (not stored in Git to keep the repository small)

```bash
python scripts/get_sample_images.py    # OpenCV sample images: 13 checkerboards, box/box_in_scene, aloe stereo pair
python scripts/get_fruits_data.py      # 8-class Fruits-360 subset (needs git) - only to re-train / re-evaluate
```

The trained model `models/fruit_cnn.keras` (3.5 MB) is shared here:
https://drive.google.com/file/d/1_fHrklYxca0EhUKS6Xq4-e42TsIA6tW6/view
Download it and put it in `models/` (next to `labels.json`) before starting the GUI or the `classify` tool.
Figures and full results are in the PDF report; the numeric results are in `outputs/*.json`.

## 3. Run everything

```bash
python scripts/run_task1_demo.py          # Task 1: calibration, filtering, edges, FFT, features, stereo -> outputs/
python -m dl.train --version v1 --epochs 25   # Task 2: baseline CNN (white background only)
python -m dl.train --version v2 --epochs 30   # Task 2: v2 stage 1 (background replacement + exposure gain + standardisation)
python -m dl.train --version v2 --epochs 12 --lr 2e-4 --finetune models/fruit_cnn_v2.keras   # stage 2 -> deployed models/fruit_cnn.keras
python -m dl.evaluate                          # Task 2: metrics, confusion matrices, robustness, Grad-CAM
python tests/test_intents.py              # Task 3: intent recognition accuracy (44 EN/VI requests)
streamlit run app/streamlit_app.py        # Task 3: the GUI (opens http://localhost:8501)
```

After downloading the model (section 2) you can start the GUI directly, training is optional.

Command-line version of the assistant:

```bash
python -m app.assistant "is this tomato ripe?" --image "data/fruits/Test/Tomato 1/10_100.jpg"
python -m app.assistant "khử nhiễu bằng median 7, add salt noise" --image data/samples/fruits.jpg --save outputs/cli
python -m app.assistant "match these two photos with ORB" --image data/features/box.png --image data/features/box_in_scene.png
```

## 4. Using the GUI

1. Upload an image (or 2 for matching/stereo, ≥3 checkerboard photos for calibration), or pick a demo set in the sidebar.
2. Type what you want in the chat box, e.g. *"what fruit is this?"*, *"show me the edges"*, *"đo kích thước quả"*,
   *"high-pass filter in the frequency domain"*. The assistant shows the detected task, confidence and parameters,
   then the result images / table. You can also force a mode with the radio buttons.

## 5. Results (short)

See `outputs/metrics.json`, `outputs/task1_results.json` and the PDF report.

## 6. Credits

* Fruits-360 dataset – H. Mureșan & M. Oltean (2018), CC BY-SA 4.0, https://github.com/Horea94/Fruit-Images-Dataset
* Checkerboard / box / aloe images – OpenCV `samples/data` (Apache 2.0); aloe stereo pair originally from the Middlebury stereo dataset.
