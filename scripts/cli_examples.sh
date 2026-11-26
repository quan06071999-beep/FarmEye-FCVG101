#!/usr/bin/env bash
# Runs a few assistant requests from the command line and stores the output for Appendix B.
cd "$(dirname "$0")/.." || exit 1
OUT=outputs/cli_examples.txt
: > "$OUT"
run () {
  echo "\$ python -m app.assistant \"$1\" ${@:2}" >> "$OUT"
  python -m app.assistant "$@" 2>/dev/null | grep -v -E "^(I0000|WARNING|To enable)" >> "$OUT"
  echo "" >> "$OUT"
}
run "is this tomato ripe?" --image "data/fruits/Test/Tomato not Ripened/$(ls 'data/fruits/Test/Tomato not Ripened' | head -1)"
run "đây là quả gì?" --image "data/fruits/Test/Lemon/$(ls data/fruits/Test/Lemon | head -1)"
run "khử nhiễu bằng median 7, add salt noise" --image data/samples/fruits.jpg
run "match these two photos with ORB, ratio 0.8" --image data/features/box.png --image data/features/box_in_scene.png
run "calibrate the camera with these checkerboards" --image data/calib/left01.jpg --image data/calib/left03.jpg --image data/calib/left04.jpg --image data/calib/left05.jpg --image data/calib/left06.jpg
run "how big is this banana? 0.6 mm/px" --image "data/fruits/Test/Banana/$(ls data/fruits/Test/Banana | head -1)"
run "estimate depth from this stereo pair" --image data/stereo/aloeL.jpg --image data/stereo/aloeR.jpg
run "hello, can you make me a coffee?" --image data/samples/fruits.jpg
cat "$OUT"
