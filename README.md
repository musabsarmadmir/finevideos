# FineVideo per-second object finder (YOLO11l)

For each second of each video, records which of five objects are present
(person, car, dog, laptop, cell phone) and how many people are visible.

## Install

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

No system packages are needed (PyAV wheels bundle FFmpeg).

## Preparation command (needs internet, run once from the repo root)

    python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"

This downloads `yolo11l.pt` into the current folder. `run.py` looks for it next
to the script, then in the current folder. Without internet the same file is at
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11l.pt

## Run

    python run.py --input <folder of .parquet files> --output results.jsonl

The default is `--stride 1`: every frame is inferred, as the brief specifies.
One worker process is started per visible GPU. Run `python run.py --help` for
the other options (`--stride`, `--batch`, `--gpus`, `--max-videos`,
`--dump-frames`, `--allow-cpu`).

The script exits with an error if `yolo11l.pt` is missing or CUDA is not
available, instead of silently running on CPU.

## How it works

- PyAV decodes each video in a background thread, so decoding overlaps with
  inference. Frames are BGR and are rotated first if the video has a rotation
  tag.
- A frame's second is `floor(timestamp - first frame's timestamp)`, using the
  frame timestamps, not frame counts.
- Frames are batched (32) within one video and run through YOLO11l with
  `imgsz=640`, `conf=0.4`, `iou=0.7`, FP16, restricted to the five classes.
  Ultralytics' own letterbox and class-aware NMS are used unchanged.
- Per second: an object is present if it is found in at least half of the
  frames used for that second. The people count is the median of the
  per-frame person counts, rounded down on a tie.
- Each Parquet file and each video has its own error handling. If a video fails
  partway, its finished seconds are kept and one error line is added. 20 failed
  videos in a row abort the run.
- Progress is saved per video in `<output>.part<N>` and `<output>.done<N>`, so
  a restarted run skips finished videos. Delete those files for a fresh run.
  At the end the parts are merged into the output file.

## The results file for the 50 training files

`results.jsonl.gz` was made with **`--stride 5`** (every 5th frame), not with
the spec-exact default. The two free Kaggle T4 GPUs I used ran at about 100
frames per second each, which is too slow to infer every frame of 125 hours in
the time available. The graded run should use the default `--stride 1`.

Checks on 8 videos (2,778 seconds, from the first two Parquet files):

| Comparison | F1 | Exact people | People within 1 |
|---|---|---|---|
| This code at stride 1 vs my earlier OpenCV every-frame FP32 run | 1.000 | 1.000 | 1.000 |
| Stride 2 vs stride 1 | 0.994 | 0.982 | 1.000 |
| Stride 3 vs stride 1 | 0.990 | 0.966 | 0.998 |
| Stride 4 vs stride 1 | 0.987 | 0.956 | 0.997 |
| Stride 5 vs stride 1 | 0.984 | 0.955 | 0.998 |

F1 is micro-averaged over the five classes (my assumption about the grader).
The sample is small, so these numbers are indicative only.

## Known limits

- The rotation direction has not been tested on a real rotated video.
- Not timed on an RTX 4090.
- The 125-hour results were produced on Kaggle with Python 3.13, torch
  2.11.0+cu130, ultralytics 8.4.216 and av 19.0.2. `requirements.txt` pins a
  CUDA 12 build of torch for the grader's driver. I could not test that
  combination, and a different torch build could change FP16 outputs slightly.
- Ultralytics 8.4.216 prints a deprecation warning about the `half` argument on
  every batch. I did not verify that FP16 is still used; measured speed was
  about 100 fps per T4.
