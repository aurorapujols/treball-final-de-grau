import pandas as pd

from pathlib import Path

from config import config

def append_rows(dataset, new_samples, are_meteors):
    print(f"APPENDING new rows: {new_samples}")
    df_new = pd.DataFrame(new_samples)
    df_new["class"] = config.labeling.meteor_label if are_meteors else config.labeling.default_label
    dataset = pd.concat([dataset, df_new], ignore_index=True)

    return dataset.drop_duplicates(subset=['filename'], keep='last')

def label_as(dataframe, filenames_list, filepath_list, label):
    print(f"Files to label: {len(filenames_list)}")
    files_to_process = []

    for (file, path) in zip(filenames_list, filepath_list):        
        rows = dataframe.loc[dataframe['filename'] == file]
        if rows.shape[0] == 0:
            print(f"⚠️ file {file} is not in the dataframe!")

            # Process it if possible
            files_to_process.append(Path(path))

        elif rows.shape[0] > 1:
            print(f"⚠️ The file {file} appears twice in the dataframe.")
        else:
            dataframe.loc[dataframe['filename'] == file, 'class'] = label

    return dataframe, files_to_process