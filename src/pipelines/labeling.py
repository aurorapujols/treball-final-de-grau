import os
import pandas as pd

from utils.archives import (
    get_filenames_to_label,
    clear_incoming_folder
)
from dataset.dataset_utils import label_as
from pipelines.preprocess_incoming import preprocess_incoming

from config import config

DATA_PATH = config.paths.data_root

INCOMING_FOLDER = config.paths.incoming
OUTPUT_FOLDER = config.paths.processed_root
RAW_FOLDER = config.paths.raw_root
DATASET = config.dataset.last_version
CSV_DATAPATH = f"{OUTPUT_FOLDER}/{DATASET}"

LABEL = config.labeling.meteor_label

def label_videos(csv_data_path=CSV_DATAPATH, label=LABEL):

    if os.path.exists(CSV_DATAPATH):
        dataset = pd.read_csv(csv_data_path, sep=";")

        print("Incoming:", get_filenames_to_label(INCOMING_FOLDER)[0])
        print("Dataset:", len(dataset))

        # Remove duplicates just in case there still are some
        dataset = dataset.drop_duplicates(subset = ['filename'], keep='last')

        filenames, filepaths = get_filenames_to_label(input_folder=INCOMING_FOLDER)
        dataset, files_to_process = label_as(dataframe=dataset, filenames_list=filenames, filepath_list=filepaths, label=label)

        print(f"Files to process: {len(files_to_process)}")
        clear_incoming_folder(path_incoming_folder=INCOMING_FOLDER, files_to_leave=files_to_process)

        # Process missing files
        preprocess_incoming(are_meteors=True)

        clear_incoming_folder(path_incoming_folder=INCOMING_FOLDER, files_to_leave=[])
        # dataset = dataset.drop_duplicates(subset = ['filename'], keep='last')
        # print(f"unique={len(list(dataset['filename'].unique()))}")
        # print(f"num_rows={dataset.shape[0]}")

        # dataset.to_csv(csv_data_path, sep=";", index=False)
        # df = pd.read_csv(CSV_DATAPATH, sep=";")
        # print("Dataset after:", len(df))

    else:
        print("⚠️ CSV doesn't exist!")


if __name__ == "__main__":
    
    print("Starting labeling...")
    label_videos()
    print("DONE!")