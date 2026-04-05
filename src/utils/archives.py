import os
import glob
import py7zr
import shutil
import subprocess
import pandas as pd

from pathlib import Path
from PIL import Image

from config import config
from xml_processing.metadata import get_bbox_metadata
from image_processing.sum_img import generate_cropped_sum_image

def extract_files_from_7z(zip_folder, output_path):
    with py7zr.SevenZipFile(zip_folder, mode='r') as archive:
        extract_path = Path(output_path).resolve()
        archive.extractall(path=str(extract_path))

def extract_files_from_multiple_7z(folder_path, remove=False):
    zip_files = glob.glob(f"{folder_path}/*.7z")
    for zf in zip_files:
        extract_files_from_7z(zip_folder=zf, output_path=folder_path)
        if remove:
            os.remove(zf)

def get_incoming_files(input_folder):
    video_files = []
    xml_files = []

    # Extract files from .7z files in input folder and delete .7z files
    extract_files_from_multiple_7z(folder_path=input_folder, remove=True)

    # Collect only the .avi and .xml files in the folder
    video_files = [str(p) for p in Path(input_folder).rglob("*.avi")]
    xml_files   = [str(p) for p in Path(input_folder).rglob("*.xml")]          

    return video_files, xml_files

def delete_files_in_folder(folder_path):
    for f in Path(folder_path).iterdir():
        if f.is_file():
            f.unlink()          # delete file
        elif f.is_dir():
            shutil.rmtree(f)    # delete folder recursively

def move_files(file_paths, dest_folder, skipped):
    dest_folder.mkdir(parents=True, exist_ok=True)

    for file_path in file_paths:
        src = Path(file_path)
        if src.stem in skipped:  
            continue    # Skip if this file was marked as skipped
        dest = dest_folder / src.name   # Always flatten: move only the file, ignore its parent folders

        dest.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent folder exists
        shutil.move(str(src), str(dest))

    # After moving, remove empty directories inside incoming
    for folder in sorted(Path(file_paths[0]).parents[1].glob("*/"), reverse=True):
        try:
            folder.rmdir()
        except OSError:
            pass


def get_xml_from_video(video_path, xml_paths):
    for xml_path in xml_paths:
        if Path(xml_path).stem == Path(video_path).stem:
            return xml_path
    return None

def folder_exists(folder_path):
    if os.path.exists(folder_path):
        print(f"✅ Folder exists: {folder_path}")
        return True
    else:
        print(f"❌ Folder does NOT exist: {folder_path}")
        return False
    
def has_videos(folder_path):
    if folder_exists(folder_path):        
        return any(Path(folder_path).rglob("*.avi"))
    
    return None

def extract_date_prefix(filename):
    """
    Extracts YYYYMM from filenames like M20250314_...
    Returns '202503' or None if malformed.
    """
    if len(filename) < 7 or not filename.startswith("M202"):
        return None
    return filename[1:7]   # e.g. M202503 → 202503

def clear_incoming_folder(path_incoming_folder, files_to_leave):

    stems_to_keep = {p.stem for p in files_to_leave}

    for f in Path(path_incoming_folder).rglob("*"):
        if f.is_file():
            if f.stem in stems_to_keep:
                print(f"\tFILE TO KEEP: {f}")
            else:
                f.unlink()

    # Remove empty directories
    for f in Path(path_incoming_folder).rglob("*"):
        if f.is_dir():
            try:
                f.rmdir()   # only deletes if empty
            except OSError:
                pass

def get_filenames_to_label(input_folder):
    # Extract all .7z archives
    zip_files = glob.glob(f"{input_folder}/*.7z")
    for zf in zip_files:
        with py7zr.SevenZipFile(zf, mode='r') as archive:
            path = Path(input_folder).resolve()
            archive.extractall(path=path)
        os.remove(zf)

    # Collect only the .avi and .xml files, delete the rest
    video_paths = [str(p) for p in Path(input_folder).rglob("*.avi")]
    video_files = [Path(p).stem for p in video_paths]

    return video_files, video_paths

def check_files_and_dataset(config):
    """
    Find the images in the folders that are not in the dataframe (the raw data is still in the .7z folders)
    """
    processed_folder = config.paths.processed_root

    # Get the list of filenames in the dataframe
    dataset_path = f"{processed_folder}/{config.dataset.not_trained}"

    dataset = pd.read_csv(dataset_path, sep=";")

    filenames = list(dataset['filename'])
    rows = []

    print("Files in the dataset:", len(filenames))
    print(filenames[:5])

    # Find files in the folder that are not in the dataset
    count = 0
    for file in os.listdir(f"{processed_folder}/sum_image"):

        name, ext = os.path.splitext(file)
        name = name[:-7]   # -12 (cropped) -7 (sum_image)
        
        if name not in filenames:
            count += 1
            print(f"({count}) Not in dataset: {name}")
            # subprocess.run(["rm", f"{processed_folder}/original/{file}"])
        else:
            rows.append(name)   # keep all the files in both the folder and dataframe

    # Find files that are in the dataset but not in the folder
    count = 0
    for file in filenames:
        if file not in rows:
            count += 1
            print(f"\n({count}) Not in folder: {file}")

def fix_deleted_original(config):
    """
    Deleted some of the original/ images, so had to extract the files and process them again to create the sum_image_cropped
    """
    processed_folder = config.paths.processed_root
    dataset_path = f"{processed_folder}/{config.dataset.not_trained}"
    dataset = pd.read_csv(dataset_path, sep=";")

    filenames = [f for f in list(dataset['filename']) if f.startswith("M202511")]
    print(len(filenames))
    files_to_process_xml = []
    files_to_process_avi = []

    count = 0
    for file in filenames:
        filename = f"{file}_CROP_SUMIMG.png"
        if not os.path.isfile(f"{processed_folder}/original/{filename}"):
            count += 1
            # print(f"File not found: {filename}")
            files_to_process_xml.append(f"november2025/{filename[:-16]}.xml")
            files_to_process_avi.append(f"november2025/{filename[:-16]}.avi")
    print(f"Not found {count} files.")

    with py7zr.SevenZipFile(f"{config.paths.raw_metadata_root}/november2025.7z", mode="r") as z:
        archive_files = z.getnames()
        print(archive_files)

    for f in files_to_process_xml:
        if not os.path.isfile(f"{config.paths.incoming}/{f}"):
            with py7zr.SevenZipFile(f"{config.paths.raw_metadata_root}/november2025.7z", mode="r") as z:
                z.extract(targets=[f], path=f"{config.paths.incoming}")
    for f in files_to_process_avi:
        if not os.path.isfile(f"{config.paths.incoming}/{f}"):
            with py7zr.SevenZipFile(f"{config.paths.raw_videos_root}/november2025.7z", mode="r") as z:
                z.extract(targets=[f], path=f"{config.paths.incoming}")

def get_cropped_image_dims(config):
    
    processed_folder = config.paths.processed_root
    dataset_path = f"{processed_folder}/{config.dataset.not_trained}"
    dataset = pd.read_csv(dataset_path, sep=";")

    col_filename = dataset.columns.get_loc("filename")
    col_width    = dataset.columns.get_loc("width")
    col_height   = dataset.columns.get_loc("height")

    for i in range(len(dataset)):
        filename = dataset.iloc[i, col_filename] + "_CROP_SUMIMG.png"
        img_path = os.path.join(config.paths.processed.original, filename)

        if os.path.isfile(img_path):
            with Image.open(img_path) as img:
                w, h = img.size

            dataset.iloc[i, col_width] = w
            dataset.iloc[i, col_height] = h
        else:
            print(f"Image {img_path} not found in folder")

    dataset.to_csv(f"{config.paths.datasets}/dataset_LAST1.csv", sep=";", index=False)

def extract_video_files(archive_path, selected_videos, extract_dir):
    total = len(selected_videos)
    extracted = 0

    for video in selected_videos:
        output_path = os.path.join(extract_dir, os.path.basename(video))

        if not os.path.isfile(output_path):
            os.makedirs(extract_dir, exist_ok=True)

            subprocess.run([
                "7z", "e", archive_path, video, f"-o{extract_dir}", "-y"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            extracted += 1

        print(f"Extracted {extracted} / {total}")

    print(f"Finished! Extracted {extracted}/{total} new videos.")

def get_images_for_3d_plotting(config):

    results_csv_path = f"{config.paths.datasets}/{config.dataset.labeling_results}"
    dataset_csv_path = f"{config.paths.datasets}/{config.dataset.not_trained}"

    results = pd.read_csv(results_csv_path, sep=";")
    dataset = pd.read_csv(dataset_csv_path, sep=";")

    results2 = pd.DataFrame(columns=results.columns)
    dataset2 = pd.DataFrame(columns=dataset.columns)

    labels = list(results['class'].unique())
    selected_files = []

    max_samples = 50

    for label in labels:

        if label == 'meteor':
            # Select from dataset
            class_df = dataset[dataset['class'] == 'meteor']
            selected = class_df.sample(n=min(max_samples, len(class_df)), random_state=42)

            selected_files.extend(selected['filename'].tolist())
            dataset2 = pd.concat([dataset2, selected], ignore_index=True)

            # Build results2 rows for meteor
            temp = pd.DataFrame({
                'filename': selected['filename'],
                'class': 'meteor'
            })
            results2 = pd.concat([results2, temp], ignore_index=True)

        else:
            # Select from results
            class_df = results[results['class'] == label]
            selected = class_df.sample(n=min(max_samples, len(class_df)), random_state=42)

            selected_files.extend(selected['filename'].tolist())
            results2 = pd.concat([results2, selected], ignore_index=True)

            # Pull matching rows from dataset
            dataset2 = pd.concat([
                dataset2,
                dataset[dataset['filename'].isin(selected['filename'])]
            ], ignore_index=True)

    # Copy images
    for f in selected_files:
        src = f"{config.paths.processed.original}/{f}_CROP_SUMIMG.png"
        dst = f"{config.paths.processed_root}/plotting"
        subprocess.run(["cp", src, dst])

    # Save CSVs
    results2.to_csv(f"{config.paths.processed_root}/plotting/results.csv", sep=";", index=False)
    dataset2.to_csv(f"{config.paths.processed_root}/plotting/dataset.csv", sep=";", index=False)