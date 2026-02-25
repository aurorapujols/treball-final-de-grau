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
DATASET_CSV = f"{DATA_PATH}/processed/dataset_36164.csv"
RESULT_CSV = "logs/labeling/results.csv"
zip_folders = ["october2025.7z", "november2025.7z", "december2025.7z"] #, "january2026.7z"]
ARCHIVE_PATH = f"{DATA_PATH}/raw_data/videos/{random.choice(zip_folders)}"
EXTRACT_DIR = f"{DATA_PATH}/labeling_tool/extracted"
N_FILES = 10

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

def get_files_to_extract(df, archive_path, filenames, needed):
    selected = []
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        video_files = [f for f in z.getnames() if f.endswith(".avi")]
        random.shuffle(video_files)
        for video in video_files:
            if len(selected) >= needed:
                break
            basename = os.path.splitext(os.path.basename(video))[0]
            if basename in filenames:
                label = df.loc[df["filename"] == basename, "class"].iloc[0]
                if label == "unknown":
                    selected.append(video)
    return selected

def extract_video_files(archive_path, selected_videos):
    total = len(selected_videos)
    progress_bar = st.progress(0)
    status_text = st.empty()
    extracted = 0

    for i, video in enumerate(selected_videos):
        output_path = os.path.join(EXTRACT_DIR, os.path.basename(video))

        if not os.path.isfile(output_path):
            os.makedirs(EXTRACT_DIR, exist_ok=True)

            subprocess.run([
                "7z", "e", archive_path, video, f"-o{EXTRACT_DIR}", "-y"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
        mp4_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

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