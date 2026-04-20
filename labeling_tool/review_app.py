import os
import py7zr
import streamlit as st
import pandas as pd
import subprocess

# -----------------------------
# CONFIG
# -----------------------------
BASE_PATH = os.path.abspath("../../../data/upftfg26/apujols")
LABELING_ROOT = f"{BASE_PATH}/results"   # <-- parent folder containing subfolders
REVIEW_DIR = f"{BASE_PATH}/review_app"
VIDEOS_PATH = f"{REVIEW_DIR}/videos"
COMMENTS_CSV = f"{REVIEW_DIR}/data/review_comments.csv"

SUMIMG_DIR    = f"{BASE_PATH}/processed/sum_image"
CROP_DIR      = f"{BASE_PATH}/processed/original"

# -----------------------------
# SESSION STATE
# -----------------------------
st.session_state.setdefault("files", [])
st.session_state.setdefault("index", 0)
st.session_state.setdefault("current_folder", None)
st.session_state.setdefault("comments", {})
st.session_state.setdefault("df_class", None)


# -----------------------------
# HELPERS
# -----------------------------

MONTH_TO_ARCHIVE = {
    (2023, 10): "oct-dec2023.7z",
    (2023, 11): "oct-dec2023.7z",
    (2023, 12): "oct-dec2023.7z",


    (2024, 1): "jan-dec2024.7z",
    (2024, 2): "jan-dec2024.7z",
    (2024, 3): "jan-dec2024.7z",
    (2024, 4): "jan-dec2024.7z",
    (2024, 5): "jan-dec2024.7z",
    (2024, 6): "jan-dec2024.7z",
    (2024, 7): "jan-dec2024.7z",
    (2024, 8): "jan-dec2024.7z",
    (2024, 9): "jan-dec2024.7z",
    (2024, 10): "jan-dec2024.7z",
    (2024, 11): "jan-dec2024.7z",
    (2024, 12): "jan-dec2024.7z",

    (2025, 1): "jan-sep2025.7z",
    (2025, 2): "jan-sep2025.7z",
    (2025, 3): "jan-sep2025.7z",
    (2025, 4): "jan-sep2025.7z",
    (2025, 5): "jan-sep2025.7z",
    (2025, 6): "jan-sep2025.7z",
    (2025, 7): "jan-sep2025.7z",
    (2025, 8): "jan-sep2025.7z",
    (2025, 9): "jan-sep2025.7z",

    (2025, 10): "october2025.7z",
    (2025, 11): "november2025.7z",
    (2025, 12): "december2025.7z",
    (2026,  1): "january2026.7z",
    (2026,  2): "february2026.7z",
}

def filename_to_archive(filename):
    try:
        digits = filename.lstrip("M")[:8]
        year  = int(digits[0:4])
        month = int(digits[4:6])
        return MONTH_TO_ARCHIVE.get((year, month))
    except:
        return None

def find_avi_in_archive(filename, archive_path):
    basename = filename
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        for internal_path in z.getnames():
            if internal_path.endswith(".avi"):
                if os.path.splitext(os.path.basename(internal_path))[0] == basename:
                    return internal_path
    return None

def convert_avi_to_mp4(avi_path):
    mp4_path = avi_path.replace(".avi", ".mp4")
    if not os.path.isfile(avi_path):
        st.error(f"AVI does not exist: {avi_path}")
        return None

    if os.path.exists(mp4_path):
        return mp4_path

    command = [
        "/home/apujols/tools/ffmpeg/ffmpeg", "-y",
        "-i", avi_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        mp4_path,
    ]

    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if os.path.isfile(mp4_path):
        return mp4_path
    else:
        st.error("FFmpeg failed to produce output.")
        return None

def get_or_extract_video(filename):
    avi_path = os.path.join(VIDEOS_PATH, filename + ".avi")
    mp4_path = os.path.join(VIDEOS_PATH, filename + ".mp4")

    # If MP4 already exists → done
    if os.path.exists(mp4_path):
        return mp4_path

    # If AVI exists but MP4 doesn't → convert
    if os.path.exists(avi_path):
        return convert_avi_to_mp4(avi_path)

    # Otherwise → extract from archive
    archive_name = filename_to_archive(filename)
    if archive_name is None:
        return None

    archive_path = os.path.join(BASE_PATH, "raw_data", "videos", archive_name)
    if not os.path.exists(archive_path):
        return None

    internal_avi = find_avi_in_archive(filename, archive_path)
    if internal_avi is None:
        return None

    # Extract AVI
    os.makedirs(VIDEOS_PATH, exist_ok=True)
    subprocess.run(
        ["7z", "e", archive_path, internal_avi, f"-o{VIDEOS_PATH}", "-y"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Convert to MP4
    if os.path.exists(avi_path):
        return convert_avi_to_mp4(avi_path)

    return None


def list_subfolders(path):
    """Return all subfolders inside a directory."""
    if not os.path.isdir(path):
        return []
    return sorted([
        f for f in os.listdir(path)
        if os.path.isdir(os.path.join(path, f))
    ])


def load_filenames(folder):
    """Load base filenames from files ending with _CROP_SUMIMG.png."""
    files = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(folder)
        if f.endswith(".png")
    ])
    return sorted(files)


def extract_video_if_needed(filename):
    """Extract AVI → MP4 into REVIEW_DIR if not already present."""
    avi_path = os.path.join(REVIEW_DIR, filename + ".avi")
    mp4_path = os.path.join(REVIEW_DIR, filename + ".mp4")

    if os.path.exists(mp4_path):
        return mp4_path

    if os.path.exists(avi_path):
        cmd = [
            "/home/apujols/tools/ffmpeg/ffmpeg", "-y",
            "-i", avi_path,
            "-vcodec", "libx264",
            "-acodec", "aac",
            mp4_path,
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return mp4_path if os.path.exists(mp4_path) else None

    return None


def parse_classifier(folder_name):
    """Extract classifier info from folder name."""
    if "1.0" in folder_name:
        return "MLP", "1.0"
    if "1.1" in folder_name:
        return "Logistic Regression", "1.1"
    return "Unknown", None


def save_comment(filename, new_comment, classifier_name):
    if not new_comment:
        return

    # 1. Load existing data or create empty
    if os.path.exists(COMMENTS_CSV):
        df = pd.read_csv(COMMENTS_CSV)
    else:
        df = pd.DataFrame(columns=["filename", "comment", "classifier"])

    # 2. Check if filename + classifier exists
    mask = (df["filename"] == filename) & (df["classifier"] == classifier_name)
    
    if mask.any():
        # Append to existing row
        old_comment = str(df.loc[mask, "comment"].iloc[0])
        if old_comment and old_comment != "nan":
            updated_text = f"{old_comment}, {new_comment}"
        else:
            updated_text = new_comment
        df.loc[mask, "comment"] = updated_text
    else:
        # Create new row
        new_row = pd.DataFrame([{
            "filename": filename,
            "comment": new_comment,
            "classifier": classifier_name
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    # 3. Save (Overwrite the whole file with the updated dataframe)
    df.to_csv(COMMENTS_CSV, index=False)


def load_classification_csv(suffix):
    csv_path = f"{REVIEW_DIR}/data/classification_results_val_{suffix}.csv"
    if not os.path.exists(csv_path):
        st.error(f"Classification CSV not found: {csv_path}")
        st.stop()
    return pd.read_csv(csv_path, sep=";")


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🔍 Misclassification Review Tool")

# List subfolders
subfolders = list_subfolders(LABELING_ROOT)

if len(subfolders) == 0:
    st.error(f"No subfolders found in {LABELING_ROOT}")
    st.stop()

selected_folder = st.selectbox(
    "Choose a folder to review:",
    subfolders,
    index=subfolders.index(st.session_state.current_folder) if st.session_state.current_folder in subfolders else 0
)

if st.button("Load folder"):
    folder_path = os.path.join(LABELING_ROOT, selected_folder)

    st.session_state.current_folder = folder_path
    st.session_state.files = load_filenames(folder_path)
    st.session_state.index = 0

    classifier_name, suffix = parse_classifier(selected_folder)
    if suffix is None:
        st.error("Folder name must contain classifier suffix (1.0 or 1.1).")
        st.stop()

    st.session_state.df_class = load_classification_csv(suffix)
    st.rerun()


# Stop if no folder loaded
if not st.session_state.current_folder:
    st.stop()

files = st.session_state.files
df_class = st.session_state.df_class

if len(files) == 0:
    st.warning("No images found in this folder.")
    st.stop()

idx = st.session_state.index
if idx >= len(files):
    st.success("Review complete!")
    st.stop()

filename = files[idx]
classifier_name, suffix = parse_classifier(st.session_state.current_folder)


# -----------------------------
# CLASSIFICATION INFO
# -----------------------------
row = df_class[df_class["filename"] == filename]
label_map = {0: "non-meteor", 1: "meteor"}

if len(row) == 1:
    y_true = int(row["y_true"].iloc[0])
    y_pred = int(row["y_pred"].iloc[0])
    p_meteor = float(row["meteor_prob"].iloc[0])
    p_non_meteor = float(row["non-meteor_prob"].iloc[0])

    true_label_str = label_map.get(y_true, "unknown")
    pred_label_str = label_map.get(y_pred, "unknown")

    if y_true == 1 and y_pred == 0:
        misclass_type = "False Negative (meteor → non-meteor)"
    elif y_true == 0 and y_pred == 1:
        misclass_type = "False Positive (non-meteor → meteor)"
    else:
        misclass_type = "Correct classification"
else:
    y_true = None
    y_pred = None
    misclass_type = "No classification info found"


# -----------------------------
# HEADER
# -----------------------------
st.subheader(f"Sample {idx+1} / {len(files)}")
st.write(f"**Filename:** {filename}")
st.write(f"**Classifier:** {classifier_name}")
st.write(f"**True label:** {true_label_str}")
st.write(f"**Predicted label:** {pred_label_str} (METEOR: {p_meteor:.4f} | NON-METEOR: {p_non_meteor})")
st.write(f"**Type:** {misclass_type}")


# -----------------------------
# Load images
# -----------------------------
sumimg_path = os.path.join(SUMIMG_DIR, filename + "_SUMIMG.png")
crop_path   = os.path.join(CROP_DIR,   filename + "_CROP_SUMIMG.png")

col1, col2 = st.columns(2)

with col1:
    if os.path.exists(sumimg_path):
        st.image(sumimg_path, caption="Sum-image")
    else:
        st.warning("[Missing]")

with col2:
    if os.path.exists(sumimg_path):
        st.image(crop_path, caption="Cropped")
    else:
        st.warning("[Missing]")


# -----------------------------
# Optional video extraction
# -----------------------------
if st.button("Show video"):
    mp4_path = get_or_extract_video(filename)
    if mp4_path and os.path.exists(mp4_path):
        st.video(mp4_path)
    else:
        st.info("No video available for this file.")


# -----------------------------
# COMMENTS
# -----------------------------
st.write("### Add a comment (optional)")

# Fetch existing comment from CSV for the current file
current_csv_comment = ""
if os.path.exists(COMMENTS_CSV):
    temp_df = pd.read_csv(COMMENTS_CSV)
    match = temp_df[(temp_df["filename"] == filename) & (temp_df["classifier"] == classifier_name)]
    if not match.empty:
        current_csv_comment = str(match["comment"].iloc[0])

# Display the existing comments (Read Only)
if current_csv_comment:
    st.info(f"**Existing notes:** {current_csv_comment}")

# Input for NEW comment only
new_comment_input = st.text_area("Add new comment (will be appended with ', '):", value="")

if st.button("Save and Append Comment"):
    if new_comment_input:
        save_comment(filename, new_comment_input, classifier_name)
        st.success("Comment updated!")
        st.rerun() # Refresh to show the updated "Existing notes"


# -----------------------------
# NAVIGATION
# -----------------------------
col_prev, col_next = st.columns(2)

with col_prev:
    if st.button("⬅️ Previous") and st.session_state.index > 0:
        st.session_state.index -= 1
        st.rerun()

with col_next:
    if st.button("Next ➡️"):
        st.session_state.index += 1
        st.rerun()


# -----------------------------
# PROGRESS BAR
# -----------------------------
st.progress(st.session_state.index / len(files))