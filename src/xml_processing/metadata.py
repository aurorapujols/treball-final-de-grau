import numpy as np
import xml.etree.ElementTree as ET

from pathlib import Path

def get_bbox_metadata(input_path, padding=32):

    # Open XML element
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Get video data
    width = int(root.attrib['cx'])
    height = int(root.attrib['cy'])
    frames = int(root.attrib['frames'])
    fps = float(root.attrib['fps'])

    # Get event occurrance time
    year = int(root.attrib['y'])
    month = int(root.attrib['mo'])
    day = int(root.attrib['d'])
    hour = int(root.attrib['h'])
    minute = int(root.attrib['m'])

    # Get location data
    lng = float(root.attrib['lng'])
    lat = float(root.attrib['lat'])
    alt = float(root.attrib['alt'])
    camera = str(root.attrib['sid'])

    # Get all uc_path elements
    path_points = root.find('ufocapture_paths')

    x_vals = [float(p.attrib['x']) for p in path_points.findall('uc_path')]
    y_vals = [float(p.attrib['y']) for p in path_points.findall('uc_path')]
    brightnes_vals = [int(p.attrib['bmax']) for p in path_points.findall('uc_path')]
    frame_nums = [int(p.attrib['fno']) for p in path_points.findall('uc_path')]

    # Compute bounding box
    bbox = {
        'x_min': max(0, min(x_vals) - padding),
        'x_max': min(width, max(x_vals) + padding),
        'y_min': max(0, min(y_vals) - padding),
        'y_max': min(height, max(y_vals) + padding)
    }

    # Compute time and brightness
    time_seconds = (frame_nums[-1] - frame_nums[0] + 1) / fps
    mean_brightness = np.mean(brightnes_vals)

    metadata = {
        "filename": Path(input_path).stem,
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "lng": lng,
        "lat": lat,
        "alt": alt,
        "camera": camera,
        "width": bbox['x_max'] - bbox['x_min'],
        "height": bbox['y_max'] - bbox['y_min'],
        "frames": frames,
        "fps": fps,
        "time": time_seconds,
        "bmin": min(brightnes_vals),
        "bmax": max(brightnes_vals),
        "mean_brightness": mean_brightness
    }

    return bbox, metadata