import os
import glob
import py7zr
import shutil

from pathlib import Path

from config import config

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