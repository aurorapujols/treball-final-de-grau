import os
import random
import subprocess

import streamlit as st
import pandas as pd
import py7zr

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "../../../data/upftfg26/apujols"
DATASET_CSV = f"{DATA_PATH}/datasets/dataset_test.csv"
RESULT_CSV = "logs/labeling/results.csv"
EXTRACT_DIR = f"{DATA_PATH}/labeling_tool/extracted"
N_FILES = 50

# Maps (year, month_number) -> archive filename
MONTH_TO_ARCHIVE = {
    (2025, 10): "october2025.7z",
    (2025, 11): "november2025.7z",
    (2025, 12): "december2025.7z",
    (2026,  1): "january2026.7z",
    (2026,  2): "february2026.7z",
}


def filename_to_archive(filename):
    """
    Map a filename like 'M20251011' to its corresponding .7z archive name.
    Returns the archive filename string, or None if the date cannot be parsed
    or the month is not in MONTH_TO_ARCHIVE.

    Expected format: M<YYYY><MM><DD>[optional suffix]
    Examples:
        M20251011  -> 'october2025.7z'
        M20260103  -> 'january2026.7z'
    """
    try:
        # Strip leading 'M' and read the first 8 digits as YYYYMMDD
        digits = filename.lstrip("M")[:8]
        year  = int(digits[0:4])
        month = int(digits[4:6])
        return MONTH_TO_ARCHIVE.get((year, month))
    except (ValueError, IndexError):
        return None


def load_dataset():
    return pd.read_csv(DATASET_CSV, sep=";")


def load_results():
    if os.path.exists(RESULT_CSV):
        return pd.read_csv(RESULT_CSV, sep=";")
    return pd.DataFrame(columns=["filename", "class"])


def save_results(new_entry):
    os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)

    if os.path.exists(RESULT_CSV):
        df = pd.read_csv(RESULT_CSV, sep=";")
    else:
        df = pd.DataFrame(columns=["filename", "class"])

    df = df[df["filename"] != new_entry["filename"]]
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(RESULT_CSV, sep=";", index=False)


def avi_exists(filename):
    avi_path = os.path.join(EXTRACT_DIR, filename + ".avi")
    return os.path.isfile(avi_path)


def get_existing_unlabeled_videos(unknown_files):
    return [f for f in unknown_files if avi_exists(f)]


def get_files_to_extract(df, videos_base_path, filenames, needed):
    """
    Select up to `needed` unknown video files from the filenames list,
    routing each filename to its correct .7z archive based on the date
    encoded in the filename (format: M<YYYY><MM><DD>...).

    Returns a dict mapping archive_path -> list of internal archive paths
    (i.e. the paths as they appear inside the .7z file).
    """
    # Pick `needed` random unknown filenames that map to a known archive
    candidates = [f for f in filenames if filename_to_archive(f) is not None]
    random.shuffle(candidates)
    selected_filenames = candidates[:needed]

    # Group selected filenames by their archive
    archive_to_filenames = {}
    for fname in selected_filenames:
        archive_name = filename_to_archive(fname)
        archive_path = os.path.join(videos_base_path, archive_name)
        archive_to_filenames.setdefault(archive_path, []).append(fname)

    # For each archive, find the exact internal path of each requested file
    archive_to_videos = {}
    for archive_path, fnames_needed in archive_to_filenames.items():
        if not os.path.isfile(archive_path):
            st.warning(f"Archive not found, skipping: {archive_path}")
            continue

        fnames_set = set(fnames_needed)
        found = []
        with py7zr.SevenZipFile(archive_path, mode="r") as z:
            for internal_path in z.getnames():
                if not internal_path.endswith(".avi"):
                    continue
                basename = os.path.splitext(os.path.basename(internal_path))[0]
                if basename in fnames_set:
                    found.append(internal_path)
                    fnames_set.discard(basename)
                    if not fnames_set:
                        break

        if found:
            archive_to_videos[archive_path] = found

    return archive_to_videos  # {archive_path: [internal_video_path, ...]}


def extract_video_files(archive_to_videos):
    """
    Extract video files given a dict of {archive_path: [internal_video_paths]}.
    Shows a combined progress bar across all files.
    """
    all_items = [
        (archive_path, video)
        for archive_path, videos in archive_to_videos.items()
        for video in videos
    ]
    total = len(all_items)
    if total == 0:
        st.warning("No videos to extract.")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    extracted = 0

    for i, (archive_path, video) in enumerate(all_items):
        output_path = os.path.join(EXTRACT_DIR, os.path.basename(video))

        if not os.path.isfile(output_path):
            os.makedirs(EXTRACT_DIR, exist_ok=True)
            subprocess.run(
                ["7z", "e", archive_path, video, f"-o{EXTRACT_DIR}", "-y"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            extracted += 1

        progress_bar.progress((i + 1) / total)
        status_text.write(f"Extracted {extracted} / {total}")

    st.success(f"Finished! Extracted {extracted}/{total} new videos.")


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


def cleanup_extracted_videos(results_df):
    """Delete extracted videos whose class is NOT 'unknown'."""
    classified = results_df[results_df["class"] != "unknown"]["filename"].tolist()

    for fname in classified:
        avi = os.path.join(EXTRACT_DIR, fname + ".avi")
        mp4 = os.path.join(EXTRACT_DIR, fname + ".mp4")

        if os.path.exists(avi):
            os.remove(avi)
        if os.path.exists(mp4):
            os.remove(mp4)
