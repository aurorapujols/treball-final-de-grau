import pandas as pd

from utils import archives

from config import config

def temp_function():

    # Change here what functions to execute
    # archives.check_files_and_dataset(config)
    archives.fix_deleted_original(config)

if __name__ == "__main__":
    
    temp_function()
    print("DONE!")