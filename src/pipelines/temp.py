import pandas as pd

from utils import archives, visualization
from dataset import dataset_utils

from config import config

def temp_function():

    # Change here what functions to execute
    archives.prepare_test_images_nightskyucp(test_dataset_path="../../../data/upftfg26/apujols/datasets/dataset_val.csv")

if __name__ == "__main__":
    
    temp_function()
    print("DONE!")