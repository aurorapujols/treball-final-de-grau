import pandas as pd

from utils import archives, visualization
from dataset import dataset_utils

from config import config

def temp_function():

    # Change here what functions to execute
    archives.get_images_for_3d_plotting(config)

if __name__ == "__main__":
    
    temp_function()
    print("DONE!")