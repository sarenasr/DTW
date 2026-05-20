# ASL Landmark Time-Series + DTW

This project extracts MediaPipe Holistic landmarks from ASL videos, converts them into time-series features, and optionally compares videos using DTW.

## What it can do

- Extract pose, left hand, right hand (and optional face) landmarks per frame from video files.
- Normalize hand landmarks to reduce signer position/scale differences (enabled by default).
- Save raw landmark arrays plus a long-format CSV for analysis.
- Build compact per-frame DTW feature matrices.
- Compare videos against a reference and record DTW distances.
- Generate time-series plots, DTW alignment plots, and side-by-side comparison plots.

## Install

```bash
python -m pip install -r requirements.txt
```

## Command

```bash
python asl_landmark_timeseries_dtw.py --input <video_or_folder> --output <output_dir> [options]
```

| Option                 | Description                                                                     |
| ---------------------- | ------------------------------------------------------------------------------- |
| `--input`              | Path to a video file or a folder containing videos (required).                  |
| `--output`             | Directory where outputs will be saved (required).                               |
| `--no-normalize-hands` | Disable wrist-centered hand normalization.                                      |
| `--include-face`       | Also save face landmarks (outputs become much larger).                          |
| `--no-csv`             | Skip writing the long-format CSV.                                               |
| `--plot`               | Save landmark time-series plots as PNG files.                                   |
| `--show-plots`         | Display plots interactively while processing.                                   |
| `--compare-to`         | Reference video or `*_dtw_features.npy` for DTW comparison.                     |
| `--plot-dtw`           | Save DTW accumulated-cost plots with alignment paths (requires `--compare-to`). |
| `--plot-compare`       | Save side-by-side landmark plots (requires `--compare-to`).                     |

Supported video extensions: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`. Folder inputs are searched recursively.

## Examples

```bash
python asl_landmark_timeseries_dtw.py --input videos --output outputs
python asl_landmark_timeseries_dtw.py --input videos/hello_01.mp4 --output outputs --plot
python asl_landmark_timeseries_dtw.py --input videos --output outputs --plot --compare-to outputs/Hello_dtw_features.npy
python asl_landmark_timeseries_dtw.py --input videos --output outputs --compare-to videos/Hello.mp4 --plot-dtw
python asl_landmark_timeseries_dtw.py --input videos/Today.mp4 --output outputs --compare-to outputs/Hello_dtw_features.npy --plot-compare
```

## Outputs

When processing a video named `<stem>.mp4`, the output directory can include:

| Output file                          | When it appears             | Description                                                             |
| ------------------------------------ | --------------------------- | ----------------------------------------------------------------------- | -------- | -------------------------------------------- | ---------------- | ------------------------------- |
| `<stem>_pose.npy`                    | Always                      | Pose landmarks per frame, shape `(T, 33, 4)`.                           |
| `<stem>_left_hand.npy`               | Always                      | Left-hand landmarks per frame, shape `(T, 21, 4)`.                      |
| `<stem>_right_hand.npy`              | Always                      | Right-hand landmarks per frame, shape `(T, 21, 4)`.                     |
| `<stem>_face.npy`                    | `--include-face`            | Face landmarks per frame, shape `(T, 468, 4)`.                          |
| `<stem>_dtw_features.npy`            | Always                      | DTW-ready feature matrix, shape `(T, D)`.                               |
| `<stem>_landmarks_long.csv`          | Default (unless `--no-csv`) | Long-format CSV: `video, frame, group, landmark_id, coordinate, value`. |
| `<stem>_left_hand_{x                 | y                           | z}\_timeseries.png`                                                     | `--plot` | Hand landmark coordinate plots (left hand).  |
| `<stem>_right_hand_{x                | y                           | z}\_timeseries.png`                                                     | `--plot` | Hand landmark coordinate plots (right hand). |
| `<stem>_pose_{x                      | y                           | z}\_timeseries.png`                                                     | `--plot` | Pose landmark coordinate plots.              |
| `<target>_vs_<reference>_dtw.png`    | `--plot-dtw`                | DTW accumulated-cost plot with alignment path.                          |
| `<target>_vs_<reference>\_{left_hand | right_hand                  | pose}\_{x                                                               | y        | z}\_side_by_side.png`                        | `--plot-compare` | Side-by-side time-series plots. |
| `dtw_comparisons.csv`                | `--compare-to`              | Aggregate DTW distances across processed videos.                        |

If `--compare-to` points to a video file, the reference video is also processed and saved into the output directory.
