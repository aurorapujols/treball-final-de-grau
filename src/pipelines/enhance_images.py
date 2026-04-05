import os
import cv2

import pandas as pd

from image_processing.enhance import (
    meteor_stretch,
    global_threshold,
    min_max_stretch,
    percentile_stretch
)

from config import config

DATA_PATH = config.paths.data_root
DATASET_PATH = f"{config.paths.datasets}/{config.dataset.not_trained}"

def run_image_enhancing(enhance_type):
    dataset = pd.read_csv(DATASET_PATH, sep=";")

    for filename in dataset['filename']:
        input_path = f"{config.paths.processed.original}/{filename}_CROP_SUMIMG.png"
        output_path = f"{config.paths.processed_root}/{enhance_type}/{filename}_CROP_ENHANCED.png"

        if not os.path.exists(output_path):
            img_greyscale_cropped = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            bmin = dataset.loc[dataset["filename"] == filename, "bmin"].iloc[0]
            bmax = dataset.loc[dataset["filename"] == filename, "bmax"].iloc[0]

            enhanced_img = None

            match enhance_type:
                case "meteors_stretch":
                    enhanced_img = meteor_stretch(img=img_greyscale_cropped, Bmin=bmin, Bmax=bmax)
                case "global_threshold":
                    enhanced_img = global_threshold(img=img_greyscale_cropped, T=bmin)
                case "min_max_stretch":
                    enhanced_img = min_max_stretch(img=img_greyscale_cropped)
                case "percentile_stretch":
                    enhanced_img = percentile_stretch(img=img_greyscale_cropped)

            success = cv2.imwrite(output_path, enhanced_img)

if __name__ == "__main__":

    img_types = ["meteors_stretch", "global_threshold", "min_max_stretch", "percentile_stretch"]

    for img_type in img_types:
        print(f"Creating {img_types} for the images.")
        run_image_enhancing(img_type)
        print(f"Finished {img_type} enhancing.")

    print("DONE!")
