import os
import pandas as pd
from pathlib import Path

from dataset.dataset_utils import append_rows
from utils.archives import (
    delete_files_in_folder, 
    move_files, 
    get_incoming_files,
    get_xml_from_video )

from image_processing.sum_img import generate_sum_image, generate_cropped_sum_image

from config import config

ARE_METEORS = False

DATA_PATH = config.paths.data_root
INCOMING_FOLDER = config.paths.incoming
OUTPUT_FOLDER = config.paths.processed_root
RAW_FOLDER = config.paths.raw_root

DATASET_FILENAME = config.dataset.last_version
DATASET_PATH = f"{OUTPUT_FOLDER}/{DATASET_FILENAME}"

def preprocess_files(avi_files, xml_files):

    new_samples = []
    skipped = []

    N = len(avi_files)
    for idx in range(1, N+1):

        if idx%10 == 0:
            print(f"Processed {idx}/{N}")
        
        curr_video_file = avi_files[idx-1]
        curr_xml_file = get_xml_from_video(curr_video_file, xml_files)

        if curr_xml_file is None:
            skipped.append(Path(curr_video_file).stem)
            print(f"⚠️ No XML found for {curr_video_file}, skipping.")
            continue
        
        sum_image = generate_sum_image(
            img_input_path=curr_video_file, 
            output_path=config.paths.processed.sum_image)

        try:
            _, _, metadata = generate_cropped_sum_image(
                sum_img=sum_image,
                img_input_path=curr_video_file, 
                xml_input_path=curr_xml_file, 
                output_path=config.paths.processed.sum_image_cropped)
            new_samples.append(metadata)
        except Exception as e:
            print(f"⚠️ Skipping {curr_xml_file}: {e}")
            skipped.append(Path(curr_xml_file).stem)
            continue

    print(f"Finished: processed {N}/{N} videos")
    return new_samples, skipped


def preprocess_incoming(are_meteors=ARE_METEORS):
    
    df = pd.DataFrame()

    # 1. Load dataset if it already exists
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH, sep=";")
        print(f"Loaded existsing dataset  with {len(df)} entries.")

    # 2. Update dataset with new rows and get files to process
    avi_files, xml_files = get_incoming_files(input_folder=INCOMING_FOLDER)

    # 3. Process videos and metadata
    if len(avi_files) > 0:
        new_samples, skipped = preprocess_files(avi_files, xml_files)
        df = append_rows(df, new_samples, are_meteors=are_meteors)

        raw_xmls_folder = Path(config.paths.raw_output.metadata)
        move_files(file_paths=xml_files, dest_folder=raw_xmls_folder, skipped=skipped)
        
        raw_avis_folder = Path(config.paths.raw_output.videos)
        move_files(file_paths=avi_files, dest_folder=raw_avis_folder, skipped=skipped)

        print(f"Skipped: [{len(skipped)}] {skipped}")

    df = df.drop_duplicates(subset=['filename'], keep='last')

    # 4. Save updated dataset
    df.to_csv(DATASET_PATH, sep=";", index=False)

    # 5. Remove everything remaining in the incoming folder
    delete_files_in_folder(INCOMING_FOLDER)

if __name__ == "__main__":

    print("Started incoming data preprocessing...")
    preprocess_incoming()
    print("DONE!")