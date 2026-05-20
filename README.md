# ASL Landmark Time-Series + DTW

Extract MediaPipe Holisitc landmarks from ASL videos, convert them to time-series features, and optionally compare videos using DTW.

## Features

- Extract pose, left/right hand (optional face) landmarks per frame
- Wrist-centered hand normalization (default)
- Save raw NumPy arrays and long-format CSV for inspection
- Build compact per-frame DTW feature matrices
- Compare videos to a reference and save DTW distances
- Produce time-series plots, DTW alignment plots, and side-by-side comparisons

## Installation

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate     # Windows (Powershell)
python -m pip install -r requirements.txt
```

## Quick usage

```bash
python asl_landmark_timeseries_dtw.py --input <video_or_folder> --output <output_dir> [options]
```

Important options:

- `--input` (required): video file or folder
- `--output` (required): output directory
- `--no-normalize-hands`: disable hand normalization
- `--include-face`: include face landmarks (large outputs)
- `--no-csv`: skip writing the long CSV
- `--plot`: save time-series PNGs
- `--show-plots`: display plots interactively
- `--compare-to`: reference video or `*_dtw_features.npy` for DTW
- `--plot-dtw`: save DTW accumulated-cost plots (needs `--compare-to`)
- `--plot-compare`: save side-by-side comparison plots (needs `--compare-to`)

Supported video extensions: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`.

## Examples

```bash
# Process a folder
python asl_landmark_timeseries_dtw.py --input videos --output outputs

# Single video with plots
python asl_landmark_timeseries_dtw.py --input videos/hello_01.mp4 --output outputs --plot

# Compare to precomputed features
python asl_landmark_timeseries_dtw.py --input videos --output outputs --compare-to outputs/Hello_dtw_features.npy --plot-dtw

# Compare to a reference video and save side-by-side plots
python asl_landmark_timeseries_dtw.py --input videos/Today.mp4 --output outputs --compare-to videos/Hello.mp4 --plot-compare
```

## Outputs

When processing `<stem>.mp4`, outputs in `<output_dir>` may include:

- `<stem>_pose.npy` — Pose array `(T, 33, 4)`
- `<stem>_left_hand.npy` — Left hand `(T, 21, 4)`
- `<stem>_right_hand.npy` — Right hand `(T, 21, 4)`
- `<stem>_face.npy` — Face `(T, 468, 4)` (if `--include-face`)
- `<stem>_dtw_features.npy` — DTW-ready matrix `(T, D)`
- `<stem>_landmarks_long.csv` — Long-format CSV (`video,frame,group,landmark_id,coordinate,value`)
- `*_timeseries.png` — Time-series plots when `--plot` is used
- `*_vs_*_dtw.png` — DTW cost/alignment plots when `--plot-dtw` is used
- `*_vs_*_..._side_by_side.png` — Side-by-side plots when `--plot-compare` is used
- `dtw_comparisons.csv` — Aggregated DTW distances (when `--compare-to` used)

If `--compare-to` points to a video, the reference is processed and saved as well.

## Contributing & Git cleanup

To avoid committing large generated files, add these to `.gitignore` (example included in repo):

```
.venv/
venv/
.venv-dtw/
outputs/
outputs.zip
videos/
__pycache__/
*.pyc
```

If large files were accidentally committed:

- Remove them from tracking (keeps them locally):

```bash
git rm -r --cached outputs videos .venv-dtw
git commit -m "Remove generated files from tracking"
```

- To fully remove them from history (use with caution), use `git filter-repo` or BFG:

```bash
# backup first
git branch backup
# then run one of the history-rewrite tools (not shown here)
```

## Notes

- For comparisons across different signers, hand normalization helps reduce scale/translation effects.
- The DTW feature matrix is a starting point — experiment with features for better results.

---

If you want additional sections (license, examples with plots, or a troubleshooting guide), say which and a one-line note about what to include.
