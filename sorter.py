from pathlib import Path
import shutil
import cv2
import numpy as np

# Configuration

# Put the sorter.py inside folder with videoes
BASE_DIR = Path(__file__).resolve().parent

# Read videos from the folder where sorter.py lives
INPUT_DIR = BASE_DIR

# Output folders will be created here:
OUT_DIR = BASE_DIR / "sorted"

# Video file extensions to process
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi"}

# How many frames to sample from each video (spread across the timeline)
SAMPLES_PER_VIDEO = 25

# Per-game thresholds for template matching score (0..1)
# CS is a bit looser than Rocket League
THRESHOLDS = {
    "cs": 0.47,
    "rocket_league": 0.55,
}

# Minimum number of sampled frames that must “hit” the threshold to accept a label
MIN_HITS_BY_LABEL = {
    "cs": 2,
    "rocket_league": 3,
}

# If any single sampled frame matches extremely well, accept immediately
STRONG_SCORE = {
    "cs": 0.70,
    "rocket_league": 0.75,
}

# Template images (relative to BASE_DIR)
# These should be tight crops of stable UI elements
TEMPLATE_PATHS = {
    "cs": [
        "templates/cs/radar.jpg",
        "templates/cs/players.jpg",
        "templates/cs/loadout.jpg",
        "templates/cs/bottomLeftHUD.jpg",
    ],
    "rocket_league": [
        "templates/rl/rlBoost1.jpg",
        "templates/rl/rlBoost2.jpg",
        "templates/rl/rlScoreAndTimer.jpg",
    ],
}


# Helpers: filesystem setup


def ensure_output_dirs() -> None:
    """
    Create output folder structure:
      sorted/cs
      sorted/rocket_league
      sorted/other games
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label in list(TEMPLATE_PATHS.keys()) + ["other games"]:
        (OUT_DIR / label).mkdir(parents=True, exist_ok=True)


# Helpers: ROI handling


def clamp_roi(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    """
    Clamp ROI coordinates so they always stay inside the frame.
    Ensures ROI has at least 1x1 size.
    """
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)

    return (x1, y1, x2, y2)


def roi_for_template(label: str, template_path: str, w: int, h: int):
    """
    Return a generous “search area” (ROI) for where each HUD element is expected.
    ROI is relative to the frame size, so it works across different resolutions.

    Notes:
    - The *template image* is a tight crop (small).
    - The ROI is a larger crop from the video frame where we search for the template.
    """
    p = template_path.lower()

    if label == "cs":
        # Radar/minimap: top-left
        if "radar" in p:
            return clamp_roi(0, 0, int(0.35 * w), int(0.40 * h), w, h)

        # Player avatars / alive players bar: top-center
        if "players" in p or "playercount" in p:
            return clamp_roi(int(0.20 * w), 0, int(0.80 * w), int(0.22 * h), w, h)

        # Bottom-left HUD: health/armor area
        if "bottomlefthud" in p or "health" in p:
            return clamp_roi(0, int(0.55 * h), int(0.45 * w), h, w, h)

        # Loadout/ammo: bottom-right
        if "loadout" in p or "ammo" in p:
            return clamp_roi(int(0.55 * w), int(0.55 * h), w, h, w, h)

        # Fallback (should rarely be used)
        return clamp_roi(0, 0, int(0.60 * w), int(0.60 * h), w, h)

    if label == "rocket_league":
        # Boost meter: bottom-right
        if "boost" in p:
            return clamp_roi(int(0.55 * w), int(0.55 * h), w, h, w, h)

        # Score + timer: top-center
        if "score" in p or "timer" in p:
            return clamp_roi(int(0.25 * w), 0, int(0.75 * w), int(0.22 * h), w, h)

        # Fallback
        return clamp_roi(int(0.40 * w), 0, w, int(0.40 * h), w, h)

    # Unknown label: search whole frame
    return clamp_roi(0, 0, w, h, w, h)


# Helpers: templates + scoring


def load_templates():
    """
    Load template images once at startup (grayscale).
    Returns:
      dict[label] -> list[(template_path_str, template_gray_np)]
    """
    loaded = {}
    for label, rel_paths in TEMPLATE_PATHS.items():
        loaded[label] = []
        for rel in rel_paths:
            abs_path = BASE_DIR / rel
            templ = cv2.imread(str(abs_path), cv2.IMREAD_GRAYSCALE)
            if templ is None:
                raise FileNotFoundError(f"Template missing/unreadable: {abs_path}")
            loaded[label].append((str(abs_path), templ))
    return loaded


def sample_frame_indices(cap: cv2.VideoCapture, n_samples: int):
    """
    Choose frame indices spread evenly across the entire video.
    """
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        return [0]
    return np.linspace(0, frame_count - 1, num=n_samples, dtype=int).tolist()


def match_score(gray_roi, templ) -> float:
    """
    Run template matching and return the best match score in [0..1] (higher is better).
    """
    res = cv2.matchTemplate(gray_roi, templ, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return float(max_val)


# Core: classify a video


def classify_video(video_path: Path, templates) -> str:
    """
    Classify a single video into:
      - "cs"
      - "rocket_league"
      - "other games"

    Strategy:
    - Sample frames across the video.
    - For each frame and each label, compute the best template match score.
    - Count “hits” when best score >= THRESHOLD[label].
    - Also track best score seen across all frames for each label.
    - Use a strong-score shortcut, otherwise use hits + best_score to decide.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "other games"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    hits = {label: 0 for label in templates.keys()}
    best_overall = {label: 0.0 for label in templates.keys()}

    for fi in sample_frame_indices(cap, SAMPLES_PER_VIDEO):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for label, templ_list in templates.items():
            best_this_frame = 0.0

            for template_path, templ in templ_list:
                x1, y1, x2, y2 = roi_for_template(label, template_path, w, h)
                roi = gray[y1:y2, x1:x2]

                # Template must be smaller than ROI for matchTemplate
                if roi.shape[0] < templ.shape[0] or roi.shape[1] < templ.shape[1]:
                    continue

                best_this_frame = max(best_this_frame, match_score(roi, templ))

            best_overall[label] = max(best_overall[label], best_this_frame)

            if best_this_frame >= THRESHOLDS.get(label, 0.5):
                hits[label] += 1

    cap.release()

    # 1) If we ever saw a very strong match, accept immediately
    for label in templates.keys():
        if best_overall[label] >= STRONG_SCORE.get(label, 1.0):
            return label

    # 2) Otherwise require minimum hits and choose best (hits first, score as tie-break)
    candidates = [
        label
        for label in templates.keys()
        if hits[label] >= MIN_HITS_BY_LABEL.get(label, 9999)
    ]
    if not candidates:
        return "other games"

    candidates.sort(key=lambda l: (hits[l], best_overall[l]), reverse=True)
    return candidates[0]


# Main: scan folder + move files


def main():
    ensure_output_dirs()
    templates = load_templates()

    for entry in INPUT_DIR.iterdir():
        if not entry.is_file():
            continue

        if entry.suffix.lower() not in VIDEO_EXTS:
            continue

        # Skip anything already inside the sorted folder
        if OUT_DIR in entry.parents:
            continue

        label = classify_video(entry, templates)
        destination = OUT_DIR / label / entry.name

        print(f"{entry.name} -> {label}")
        shutil.move(str(entry), str(destination))


if __name__ == "__main__":
    main()
