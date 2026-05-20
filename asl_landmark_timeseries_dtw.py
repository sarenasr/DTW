"""
ASL / dynamic sign landmark extraction for DTW analysis
-------------------------------------------------------

What this script does:
1. Reads one video or a folder of videos.
2. Uses MediaPipe Holistic to extract pose, left hand, right hand, and face landmarks per frame.
3. Converts landmarks into time-series features.
4. Saves:
   - Raw landmark time series as .npy
   - Long-format CSV for inspection
   - Summary plots of selected landmark coordinates over time
5. Optionally normalizes hand landmarks to reduce differences caused by signer position or scale.
6. Optionally compares each processed video to a reference with DTW and saves distances (and plots).
7. Optionally saves side-by-side landmark plots for visual comparison.

Recommended folder structure:

project/
  videos/
    hello_01.mp4
    hello_02.mp4
    thankyou_01.mp4
  outputs/
  asl_landmark_timeseries_dtw.py

Install:
  pip install opencv-python mediapipe numpy pandas matplotlib tqdm

Run examples:
  python asl_landmark_timeseries_dtw.py --input videos --output outputs
  python asl_landmark_timeseries_dtw.py --input videos/hello_01.mp4 --output outputs --plot
  python asl_landmark_timeseries_dtw.py --input videos --output outputs --plot --compare-to outputs/Hello_dtw_features.npy
  python asl_landmark_timeseries_dtw.py --input videos --output outputs --compare-to videos/Hello.mp4 --plot-dtw
  python asl_landmark_timeseries_dtw.py --input videos/Today.mp4 --output outputs --compare-to outputs/Hello_dtw_features.npy --plot-compare
  python dtw_compare_library.py --reference outputs/Hello_dtw_features.npy --targets outputs/Today_dtw_features.npy --plot-dtw

Notes for DTW:
- For early experiments, hand landmarks are usually more useful than face landmarks.
- Use normalized hand coordinates for comparing the same sign across different people/videos.
- For signs involving torso/arm movement, include pose landmarks too.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# MediaPipe landmark counts
N_POSE = 33
N_HAND = 21
N_FACE = 468


# A smaller, more DTW-friendly feature subset.
# You can expand this later.
HAND_LANDMARKS_OF_INTEREST = {
    0: "wrist",
    4: "thumb_tip",
    8: "index_tip",
    12: "middle_tip",
    16: "ring_tip",
    20: "pinky_tip",
}

POSE_LANDMARKS_OF_INTEREST = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
}


def list_videos(input_path: Path) -> List[Path]:
    """Return a sorted list of video files from a file or directory."""
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {input_path.suffix}")
        return [input_path]

    if input_path.is_dir():
        videos = [p for p in input_path.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]
        return sorted(videos)

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def landmarks_to_array(landmarks, n_landmarks: int) -> np.ndarray:
    """
    Convert MediaPipe landmarks to an array of shape:
        (n_landmarks, 4)
    columns:
        x, y, z, visibility

    For hands and face, MediaPipe does not always provide visibility.
    In those cases, visibility is set to 1.0.
    Missing landmarks are filled with NaN.
    """
    arr = np.full((n_landmarks, 4), np.nan, dtype=np.float32)

    if landmarks is None:
        return arr

    for i, lm in enumerate(landmarks.landmark):
        if i >= n_landmarks:
            break
        arr[i, 0] = lm.x
        arr[i, 1] = lm.y
        arr[i, 2] = lm.z
        arr[i, 3] = getattr(lm, "visibility", 1.0)

    return arr


def normalize_hand(hand_arr: np.ndarray) -> np.ndarray:
    """
    Normalize one hand's landmarks for one frame.

    Input shape:
        (21, 4), columns x, y, z, visibility

    Normalization:
    - Translate so wrist landmark 0 is at the origin.
    - Scale by distance between wrist and middle finger MCP landmark 9.

    Why this matters:
    DTW should compare motion patterns, not simply video position or hand size.
    """
    normalized = hand_arr.copy()

    if np.isnan(normalized[:, :3]).all():
        return normalized

    wrist = normalized[0, :3]
    middle_mcp = normalized[9, :3]

    if np.isnan(wrist).any() or np.isnan(middle_mcp).any():
        return normalized

    scale = np.linalg.norm(middle_mcp - wrist)
    if scale < 1e-6:
        scale = 1.0

    normalized[:, :3] = (normalized[:, :3] - wrist) / scale
    return normalized


def extract_video_landmarks(
    video_path: Path,
    normalize_hands: bool = True,
    include_face: bool = False,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Extract landmarks from a video using MediaPipe Holistic.

    Returns a dictionary containing arrays:
        pose:       (T, 33, 4)
        left_hand:  (T, 21, 4)
        right_hand: (T, 21, 4)
        face:       (T, 468, 4), optional

    T = number of frames successfully read.
    """
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    pose_frames = []
    left_hand_frames = []
    right_hand_frames = []
    face_frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as holistic:
        pbar = tqdm(total=total_frames, desc=f"Extracting {video_path.name}")

        while True:
            success, frame_bgr = cap.read()
            if not success:
                break

            # MediaPipe expects RGB.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = holistic.process(frame_rgb)

            pose_arr = landmarks_to_array(results.pose_landmarks, N_POSE)
            left_arr = landmarks_to_array(results.left_hand_landmarks, N_HAND)
            right_arr = landmarks_to_array(results.right_hand_landmarks, N_HAND)

            if normalize_hands:
                left_arr = normalize_hand(left_arr)
                right_arr = normalize_hand(right_arr)

            pose_frames.append(pose_arr)
            left_hand_frames.append(left_arr)
            right_hand_frames.append(right_arr)

            if include_face:
                face_arr = landmarks_to_array(results.face_landmarks, N_FACE)
                face_frames.append(face_arr)

            pbar.update(1)

        pbar.close()

    cap.release()

    output = {
        "pose": np.stack(pose_frames, axis=0),
        "left_hand": np.stack(left_hand_frames, axis=0),
        "right_hand": np.stack(right_hand_frames, axis=0),
    }

    if include_face:
        output["face"] = np.stack(face_frames, axis=0)

    return output


def save_landmarks_npy(landmarks: Dict[str, np.ndarray], output_dir: Path, stem: str) -> None:
    """Save each landmark group as a separate .npy file."""
    for group_name, arr in landmarks.items():
        np.save(output_dir / f"{stem}_{group_name}.npy", arr)


def landmarks_to_long_dataframe(
    landmarks: Dict[str, np.ndarray],
    video_name: str,
) -> pd.DataFrame:
    """
    Convert landmark arrays to a long-format dataframe:

    video | frame | group | landmark_id | coordinate | value

    This is easy to inspect, filter, and plot.
    """
    rows = []

    for group_name, arr in landmarks.items():
        # arr shape: (T, L, 4)
        T, L, C = arr.shape
        coord_names = ["x", "y", "z", "visibility"]

        for frame_idx in range(T):
            for landmark_id in range(L):
                for coord_idx in range(C):
                    rows.append(
                        {
                            "video": video_name,
                            "frame": frame_idx,
                            "group": group_name,
                            "landmark_id": landmark_id,
                            "coordinate": coord_names[coord_idx],
                            "value": arr[frame_idx, landmark_id, coord_idx],
                        }
                    )

    return pd.DataFrame(rows)


def make_dtw_feature_matrix(
    landmarks: Dict[str, np.ndarray],
    use_pose: bool = True,
    use_hands: bool = True,
    selected_coordinates: Tuple[int, ...] = (0, 1, 2),
) -> np.ndarray:
    """
    Create a compact feature matrix for DTW.

    Output shape:
        (T, D)

    T = number of frames
    D = selected flattened landmark coordinates

    This is the array you can feed into multivariate DTW.
    Missing values are replaced by 0.0 by default.

    Important:
    This is a starting point, not a final research-grade representation.
    For serious experiments, you should test multiple feature sets.
    """
    feature_blocks = []

    if use_hands:
        for group_name in ["left_hand", "right_hand"]:
            arr = landmarks[group_name][:, :, selected_coordinates]
            feature_blocks.append(arr.reshape(arr.shape[0], -1))

    if use_pose:
        arr = landmarks["pose"][:, :, selected_coordinates]
        feature_blocks.append(arr.reshape(arr.shape[0], -1))

    if not feature_blocks:
        raise ValueError("No feature groups selected.")

    features = np.concatenate(feature_blocks, axis=1)
    features = np.nan_to_num(features, nan=0.0)
    return features.astype(np.float32)


def compute_dtw_accumulated_cost(
    reference: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """
    Compute the accumulated DTW cost matrix.

    Shapes:
        reference: (T_ref, D)
        target:    (T_tgt, D)
    """
    if reference.ndim != 2 or target.ndim != 2:
        raise ValueError("DTW inputs must be 2D arrays: (time, features).")
    if reference.shape[1] != target.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: {reference.shape[1]} vs {target.shape[1]}"
        )
    if reference.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("DTW inputs must have at least one frame.")

    ref_len, tgt_len = reference.shape[0], target.shape[0]
    acc = np.full((ref_len, tgt_len), np.inf, dtype=np.float32)

    def frame_cost(i: int, j: int) -> float:
        return float(np.linalg.norm(reference[i] - target[j]))

    acc[0, 0] = frame_cost(0, 0)
    for i in range(1, ref_len):
        acc[i, 0] = acc[i - 1, 0] + frame_cost(i, 0)
    for j in range(1, tgt_len):
        acc[0, j] = acc[0, j - 1] + frame_cost(0, j)

    for i in range(1, ref_len):
        for j in range(1, tgt_len):
            acc[i, j] = frame_cost(i, j) + min(
                acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]
            )

    return acc


def backtrack_dtw_path(acc_cost: np.ndarray) -> np.ndarray:
    """Backtrack the optimal DTW path from an accumulated cost matrix."""
    if acc_cost.ndim != 2:
        raise ValueError("DTW accumulated cost must be a 2D array.")

    i, j = acc_cost.shape[0] - 1, acc_cost.shape[1] - 1
    path = [(i, j)]

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = np.argmin(
                [acc_cost[i - 1, j], acc_cost[i, j - 1], acc_cost[i - 1, j - 1]]
            )
            if step == 0:
                i -= 1
            elif step == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))

    path.reverse()
    return np.array(path, dtype=np.int32)


def plot_dtw_alignment(
    acc_cost: np.ndarray,
    path: np.ndarray,
    output_path: Path,
    reference_label: str,
    target_label: str,
) -> None:
    """Save a DTW accumulated-cost heatmap with the alignment path."""
    plt.figure(figsize=(7, 6))
    plt.imshow(acc_cost.T, origin="lower", aspect="auto", cmap="magma")
    plt.plot(path[:, 0], path[:, 1], color="cyan", linewidth=1.0)
    plt.xlabel(f"Reference frames ({reference_label})")
    plt.ylabel(f"Target frames ({target_label})")
    plt.title(f"DTW alignment: {target_label} vs {reference_label}")
    plt.colorbar(label="Accumulated cost")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def load_or_extract_reference_features(
    compare_to: Path,
    output_dir: Path,
    normalize_hands: bool,
    include_face: bool,
) -> Tuple[np.ndarray, str]:
    """Load reference DTW features from .npy or extract them from a video."""
    if not compare_to.exists():
        raise FileNotFoundError(f"Reference path does not exist: {compare_to}")

    if compare_to.suffix.lower() == ".npy":
        features = np.load(compare_to)
        ref_label = compare_to.stem
        if ref_label.endswith("_dtw_features"):
            ref_label = ref_label[: -len("_dtw_features")]
        return features, ref_label

    if compare_to.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("Reference must be a video file or *_dtw_features.npy.")

    landmarks = extract_video_landmarks(
        video_path=compare_to,
        normalize_hands=normalize_hands,
        include_face=include_face,
    )
    features = make_dtw_feature_matrix(landmarks, use_pose=True, use_hands=True)
    np.save(output_dir / f"{compare_to.stem}_dtw_features.npy", features)
    return features, compare_to.stem


def load_saved_landmarks(
    output_dir: Path,
    stem: str,
    include_face: bool,
) -> Dict[str, np.ndarray]:
    groups = ["pose", "left_hand", "right_hand"]
    if include_face:
        groups.append("face")

    landmarks = {}
    for group in groups:
        path = output_dir / f"{stem}_{group}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing saved landmarks: {path}. Re-run extraction with --input {stem}.mp4"
            )
        landmarks[group] = np.load(path)

    return landmarks


def plot_landmark_timeseries(
    landmarks: Dict[str, np.ndarray],
    output_dir: Path,
    stem: str,
    show: bool = False,
) -> None:
    """
    Generate practical plots for visual comparison.

    Plots:
    1. Hand fingertip x/y trajectories over time.
    2. Pose wrist/elbow/shoulder x/y trajectories over time.
    """
    plot_hand_group(landmarks, "left_hand", output_dir, stem, show=show)
    plot_hand_group(landmarks, "right_hand", output_dir, stem, show=show)
    plot_pose_group(landmarks, output_dir, stem, show=show)


def plot_hand_group(
    landmarks: Dict[str, np.ndarray],
    group_name: str,
    output_dir: Path,
    stem: str,
    show: bool = False,
) -> None:
    arr = landmarks[group_name]
    frames = np.arange(arr.shape[0])

    for coord_idx, coord_name in [(0, "x"), (1, "y")]:
        plt.figure(figsize=(12, 6))

        for landmark_id, label in HAND_LANDMARKS_OF_INTEREST.items():
            values = arr[:, landmark_id, coord_idx]
            plt.plot(frames, values, label=f"{landmark_id}_{label}_{coord_name}")

        plt.title(f"{stem} - {group_name} {coord_name} over time")
        plt.xlabel("Frame")
        plt.ylabel(f"Normalized {coord_name}" if group_name in ["left_hand", "right_hand"] else coord_name)
        plt.legend(loc="best")
        plt.tight_layout()

        path = output_dir / f"{stem}_{group_name}_{coord_name}_timeseries.png"
        plt.savefig(path, dpi=150)
        if show:
            plt.show()
        plt.close()


def plot_pose_group(
    landmarks: Dict[str, np.ndarray],
    output_dir: Path,
    stem: str,
    show: bool = False,
) -> None:
    arr = landmarks["pose"]
    frames = np.arange(arr.shape[0])

    for coord_idx, coord_name in [(0, "x"), (1, "y")]:
        plt.figure(figsize=(12, 6))

        for landmark_id, label in POSE_LANDMARKS_OF_INTEREST.items():
            values = arr[:, landmark_id, coord_idx]
            plt.plot(frames, values, label=f"{landmark_id}_{label}_{coord_name}")

        plt.title(f"{stem} - pose {coord_name} over time")
        plt.xlabel("Frame")
        plt.ylabel(coord_name)
        plt.legend(loc="best")
        plt.tight_layout()

        path = output_dir / f"{stem}_pose_{coord_name}_timeseries.png"
        plt.savefig(path, dpi=150)
        if show:
            plt.show()
        plt.close()


def plot_compare_group(
    reference: np.ndarray,
    target: np.ndarray,
    group_name: str,
    output_dir: Path,
    reference_label: str,
    target_label: str,
    landmark_map: Dict[int, str],
) -> None:
    for coord_idx, coord_name in [(0, "x"), (1, "y")]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        ref_frames = np.arange(reference.shape[0])
        tgt_frames = np.arange(target.shape[0])

        for landmark_id, label in landmark_map.items():
            axes[0].plot(ref_frames, reference[:, landmark_id, coord_idx], label=f"{landmark_id}_{label}")
            axes[1].plot(tgt_frames, target[:, landmark_id, coord_idx], label=f"{landmark_id}_{label}")

        axes[0].set_title(f"{reference_label} - {group_name} {coord_name}")
        axes[1].set_title(f"{target_label} - {group_name} {coord_name}")
        axes[0].set_xlabel("Frame")
        axes[1].set_xlabel("Frame")

        y_label = (
            f"Normalized {coord_name}"
            if group_name in ["left_hand", "right_hand"]
            else coord_name
        )
        axes[0].set_ylabel(y_label)

        axes[0].legend(loc="best")
        axes[1].legend(loc="best")

        fig.tight_layout()
        path = output_dir / f"{target_label}_vs_{reference_label}_{group_name}_{coord_name}_side_by_side.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_side_by_side_timeseries(
    reference_landmarks: Dict[str, np.ndarray],
    target_landmarks: Dict[str, np.ndarray],
    output_dir: Path,
    reference_label: str,
    target_label: str,
) -> None:
    plot_compare_group(
        reference=reference_landmarks["left_hand"],
        target=target_landmarks["left_hand"],
        group_name="left_hand",
        output_dir=output_dir,
        reference_label=reference_label,
        target_label=target_label,
        landmark_map=HAND_LANDMARKS_OF_INTEREST,
    )
    plot_compare_group(
        reference=reference_landmarks["right_hand"],
        target=target_landmarks["right_hand"],
        group_name="right_hand",
        output_dir=output_dir,
        reference_label=reference_label,
        target_label=target_label,
        landmark_map=HAND_LANDMARKS_OF_INTEREST,
    )
    plot_compare_group(
        reference=reference_landmarks["pose"],
        target=target_landmarks["pose"],
        group_name="pose",
        output_dir=output_dir,
        reference_label=reference_label,
        target_label=target_label,
        landmark_map=POSE_LANDMARKS_OF_INTEREST,
    )


def process_video(
    video_path: Path,
    output_dir: Path,
    normalize_hands: bool,
    include_face: bool,
    save_csv: bool,
    plot: bool,
    show_plots: bool,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Extract, save, and optionally plot one video."""
    stem = video_path.stem

    landmarks = extract_video_landmarks(
        video_path=video_path,
        normalize_hands=normalize_hands,
        include_face=include_face,
    )

    save_landmarks_npy(landmarks, output_dir, stem)

    dtw_features = make_dtw_feature_matrix(
        landmarks,
        use_pose=True,
        use_hands=True,
        selected_coordinates=(0, 1, 2),
    )
    np.save(output_dir / f"{stem}_dtw_features.npy", dtw_features)

    if save_csv:
        df = landmarks_to_long_dataframe(landmarks, video_name=video_path.name)
        df.to_csv(output_dir / f"{stem}_landmarks_long.csv", index=False)

    if plot:
        plot_landmark_timeseries(landmarks, output_dir, stem, show=show_plots)

    print(f"Done: {video_path.name}")
    print(f"  Frames: {dtw_features.shape[0]}")
    print(f"  DTW feature dimensions per frame: {dtw_features.shape[1]}")
    return dtw_features, landmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe landmarks from ASL videos for time-series and DTW analysis."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to a video file or folder containing videos.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Directory where outputs will be saved.",
    )
    parser.add_argument(
        "--no-normalize-hands",
        action="store_true",
        help="Disable wrist-centered hand normalization.",
    )
    parser.add_argument(
        "--include-face",
        action="store_true",
        help="Also save face landmarks. This makes outputs much larger.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not save long-format CSV files.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save landmark time-series plots as PNG files.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively while processing.",
    )
    parser.add_argument(
        "--compare-to",
        type=str,
        help="Reference video or *_dtw_features.npy to compare with DTW.",
    )
    parser.add_argument(
        "--plot-dtw",
        action="store_true",
        help="Save DTW accumulated-cost plots with alignment paths.",
    )
    parser.add_argument(
        "--plot-compare",
        action="store_true",
        help="Save side-by-side landmark plots when using --compare-to.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(input_path)
    if not videos:
        raise RuntimeError(f"No videos found in: {input_path}")

    print(f"Found {len(videos)} video(s).")

    reference_features = None
    reference_label = None
    reference_landmarks = None
    if args.compare_to:
        compare_to_path = Path(args.compare_to)
        if not compare_to_path.exists():
            raise FileNotFoundError(f"Reference path does not exist: {compare_to_path}")
        if args.plot_compare and compare_to_path.suffix.lower() in VIDEO_EXTENSIONS:
            reference_landmarks = extract_video_landmarks(
                video_path=compare_to_path,
                normalize_hands=not args.no_normalize_hands,
                include_face=args.include_face,
            )
            reference_features = make_dtw_feature_matrix(
                reference_landmarks, use_pose=True, use_hands=True
            )
            save_landmarks_npy(reference_landmarks, output_dir, compare_to_path.stem)
            np.save(output_dir / f"{compare_to_path.stem}_dtw_features.npy", reference_features)
            reference_label = compare_to_path.stem
        else:
            reference_features, reference_label = load_or_extract_reference_features(
                compare_to_path,
                output_dir=output_dir,
                normalize_hands=not args.no_normalize_hands,
                include_face=args.include_face,
            )
            if args.plot_compare:
                reference_landmarks = load_saved_landmarks(
                    output_dir=output_dir,
                    stem=reference_label,
                    include_face=args.include_face,
                )
    elif args.plot_compare:
        raise ValueError("--plot-compare requires --compare-to.")

    comparisons = []
    for video_path in videos:
        dtw_features, landmarks = process_video(
            video_path=video_path,
            output_dir=output_dir,
            normalize_hands=not args.no_normalize_hands,
            include_face=args.include_face,
            save_csv=not args.no_csv,
            plot=args.plot,
            show_plots=args.show_plots,
        )

        if reference_features is not None:
            acc_cost = compute_dtw_accumulated_cost(reference_features, dtw_features)
            dtw_distance = float(acc_cost[-1, -1])
            comparisons.append(
                {
                    "reference": reference_label,
                    "target": video_path.stem,
                    "dtw_distance": dtw_distance,
                    "reference_frames": reference_features.shape[0],
                    "target_frames": dtw_features.shape[0],
                }
            )

            if args.plot_dtw:
                path = backtrack_dtw_path(acc_cost)
                plot_dtw_alignment(
                    acc_cost,
                    path,
                    output_dir / f"{video_path.stem}_vs_{reference_label}_dtw.png",
                    reference_label=reference_label,
                    target_label=video_path.stem,
                )
            if args.plot_compare and reference_landmarks is not None:
                plot_side_by_side_timeseries(
                    reference_landmarks=reference_landmarks,
                    target_landmarks=landmarks,
                    output_dir=output_dir,
                    reference_label=reference_label,
                    target_label=video_path.stem,
                )

    if comparisons:
        df = pd.DataFrame(comparisons)
        output_path = output_dir / "dtw_comparisons.csv"
        write_header = not output_path.exists()
        df.to_csv(output_path, mode="a", index=False, header=write_header)
        action = "Created" if write_header else "Appended to"
        print(f"{action} DTW comparisons at: {output_path}")


if __name__ == "__main__":
    main()
