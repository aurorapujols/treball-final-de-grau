import os
import random
import subprocess

import streamlit as st
import pandas as pd
import py7zr

from labeling_utils import ( 
    load_dataset, load_results, save_results, 
    avi_exists, get_existing_unlabeled_videos, get_files_to_extract, extract_video_files, 
    convert_avi_to_mp4, cleanup_extracted_videos, 
    DATA_PATH, ARCHIVE_PATH, EXTRACT_DIR, N_FILES)

zip_folders = ["october2025.7z", "november2025.7z", "december2025.7z", "january2026.7z", "february2026.7z"]     # other folders only have meteors

# -----------------------------
# SESSION STATE
# -----------------------------
st.session_state.setdefault("ready_to_label", False) 
st.session_state.setdefault("extracted", False) 
st.session_state.setdefault("sample_files", []) 
st.session_state.setdefault("index", 0)

# -----------------------------
# STREAMLIT APP
# -----------------------------

st.set_page_config(layout="wide")
st.title("🎥 Video Classification Tool")

dataset = load_dataset()
results = load_results()

st.write("### Mode selection")
mode = st.radio("Choose labeling mode:",
                ["Classify new unknown videos",
                 "Reclassify previously labeled unknown videos"
                ])

if mode == "Reclassify previously labeled unknwon videos":
    reclassify_files = results[results["class"] == "unknown"]["filename"].tolist()
    st.info(f"Found {len(reclassify_files)} videos previously labeled as 'unknown'.")
    unknown_files = reclassify_files

else:
    merged = dataset.merge(results, on="filename", how="left", suffixes=("", "_labeled"))
    remaining = merged[merged["class_labeled"].isna()]
    unknown_files = remaining[remaining["class"] == "unknown"]["filename"].tolist()

st.write(f"Videos remaining to classify: **{len(unknown_files)}**")

# -----------------------------
# EXTRACTION PHASE
# -----------------------------
if not st.session_state.ready_to_label:

    if mode == "Reclassify previously labeled unknown videos":
        reclassify_files = results[results["class"] == "unknown"]["filename"].tolist()
        existing_unlabeled = get_existing_unlabeled_videos(reclassify_files)

        st.info(f"Found {len(reclassify_files)} videos previously labeled as 'unknown'.")
        st.write(f"Videos available for reclassification: **{len(existing_unlabeled)}**")

        if len(existing_unlabeled) == 0:
            st.error("No reclassifiable videos are currently extracted.")
            st.stop()

        if st.button("Start reclassification"):
            st.session_state.sample_files = existing_unlabeled
            st.session_state.ready_to_label = True
            st.session_state.index = 0
            st.rerun()

        st.stop()  # hard stop, prevents other mode UI

    elif mode == "Classify new unknown videos":
        archive_path = f"{DATA_PATH}/raw_data/videos/{random.choice(zip_folders)}"
        existing_unlabeled = get_existing_unlabeled_videos(unknown_files)

        if len(existing_unlabeled) >= N_FILES:
            st.info(f"Using {N_FILES} already extracted videos.")
            if st.button("Start labeling"):
                st.session_state.sample_files = random.sample(existing_unlabeled, N_FILES)
                st.session_state.ready_to_label = True
                st.session_state.index = 0
                st.rerun()
        else:
            needed = N_FILES - len(existing_unlabeled)
            st.info(f"{len(existing_unlabeled)} videos already extracted. Need {needed} more.")

            not_extracted = [f for f in unknown_files if f not in existing_unlabeled]
            
            selected_videos = get_files_to_extract(dataset, archive_path, not_extracted, needed)

            if not st.session_state.extracted:
                if st.button("Extract missing videos"):
                    extract_video_files(archive_path, selected_videos)
                    st.session_state.extracted = True
            else:
                st.info("Videos already extracted.")

            existing_unlabeled = get_existing_unlabeled_videos(unknown_files)

            if len(existing_unlabeled) >= N_FILES:
                if st.button("Start labeling"):
                    st.session_state.sample_files = random.sample(existing_unlabeled, N_FILES)
                    st.session_state.ready_to_label = True
                    st.session_state.index = 0

        st.stop()


# -----------------------------
# LABELING PHASE
# -----------------------------

sample_files = st.session_state.sample_files

if len(sample_files) == 0:
    st.error("No videos available to classify.")
    st.stop()

if st.session_state.index >= len(sample_files):
    st.success("Batch complete!")
    cleanup_extracted_videos(results)
    st.stop()

current_file = sample_files[st.session_state.index]

avi_path = os.path.join(EXTRACT_DIR, current_file + ".avi")
mp4_path = convert_avi_to_mp4(avi_path)

st.subheader(f"Video {st.session_state.index + 1} / {len(sample_files)}")

SUMIMG_DIR = f"{DATA_PATH}/processed/sum_image"
CROP_DIR = f"{DATA_PATH}/processed/original"
WATERMARK_DIR = f"{DATA_PATH}/processed/global_threshold"
sumimg_path = os.path.join(SUMIMG_DIR, current_file + "_SUMIMG.png")
crop_path = os.path.join(CROP_DIR, current_file + "_CROP_SUMIMG.png")
watermark_path = os.path.join(WATERMARK_DIR, current_file + "_CROP_ENHANCED.png")

col1, col2, col3 = st.columns([1,1,0.5])

with col1:
    if mp4_path and os.path.isfile(mp4_path):
        st.video(mp4_path)
    else:
        st.error(f"Video not found or conversion failed: {current_file}")

with col2:
    if os.path.isfile(sumimg_path):
        st.image(sumimg_path, caption="Sum-image Image")
    else:
        st.warning(f"No summary image found for {sumimg_path}")

with col3:
    if os.path.isfile(crop_path):
        st.image(crop_path, caption="Cropped Image")
    else:
        st.warning(f"No cropped image found for {crop_path}")

st.write("### Assign a class")

# Dynamic class list
if "classes" not in st.session_state:
    base_classes = sorted(results["class"].unique())
    base_classes = [c for c in base_classes if c != "unknown"]
    st.session_state.classes = base_classes

new_class = st.text_input("Or create a new class")

if st.button("Add class"):
    if new_class.strip() and new_class not in st.session_state.classes:
        st.session_state.classes.append(new_class.strip())
        st.success(f"Added new class: {new_class}")

selected_class = st.selectbox("Choose class", ["unknown"] + st.session_state.classes)

if st.button("Save classification"):
    final_class = new_class.strip() if new_class.strip() else selected_class

    entry = {"filename": current_file, "class": final_class}
    save_results(entry)

    st.success(f"Saved: {current_file} → {final_class}")

    st.session_state.index += 1
    st.rerun()

progress = st.session_state.index / len(sample_files)
st.progress(progress)
