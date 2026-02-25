import cv2
import numpy as np

from pathlib import Path

from xml_processing.metadata import get_bbox_metadata

from config import config

pp_config = config.preprocessing

def apply_mask(frame, mask):
    """
    ``apply_mask`` applies a the binary mask `mask` to the given `frame` of a video.

    :param frame: frame of a video in `cv2.COLOR_BGR2GRAY` format
    :param mask: binary mask with same size as `frame`
    :return: returns the frame image with the mask applied
    """ 
    return cv2.bitwise_and(frame, frame, mask=mask)

def remove_black_frame(sum_image,
                       top = pp_config.crop_black_frame.top, 
                       bottom = pp_config.crop_black_frame.bottom, 
                       left = pp_config.crop_black_frame.left, 
                       right = pp_config.crop_black_frame.right):
    h, w = sum_image.shape
    return sum_image[top : h - bottom, left : w - right]

def generate_sum_image(img_input_path, output_path):

    # STEP 1: Read the video -----------------------------------------------------------------
    cap = cv2.VideoCapture(img_input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {img_input_path}")
    
    # ---- Read the properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # STEP 2: Define a mask ------------------------------------------------------------------
    my_mask = np.zeros((height, width), dtype=np.uint8)
    mask_height = height - config.preprocessing.mask_pixels
    cv2.rectangle(my_mask, (0, 0), (width, mask_height), 255, thickness=-1)

    # STEP 3: Read first frame and perpare background ----------------------------------------
    ret, first_frame = cap.read()
    if not ret:
        raise IOError("Cannot read first frame")
    
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    masked_first_frame = apply_mask(first_frame_gray, my_mask)

    # STEP 4: Process frames -----------------------------------------------------------------
    frame_index = 0

    sum_image = np.zeros_like(masked_first_frame, dtype=np.uint8)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1    # Increment frame that we are processing

        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # Grayscaled first frame
        current_frame_masked = apply_mask(current_frame_gray, my_mask)
        frame_diff = cv2.absdiff(current_frame_masked, masked_first_frame)    # Subtract first frame to remove background
        sum_image = np.maximum(sum_image, frame_diff)     # Add frame to the sum-image (getting the max value per pixel)

    sum_image = remove_black_frame(sum_image)
    out_path = Path(output_path) / f"{Path(img_input_path).stem}_SUMIMG.png"
    cv2.imwrite(str(out_path), sum_image)

    cap.release()

    return sum_image

def generate_cropped_sum_image(sum_img, img_input_path, xml_input_path, output_path, 
                               padding=pp_config.bbox_padding, 
                               top=pp_config.bbox_adjust.top_offset, 
                               left=pp_config.bbox_adjust.left_offset):

    # Get bounding box and crop sum-image
    bbox, metadata = get_bbox_metadata(input_path=xml_input_path, padding=padding)

    x_min, x_max = int(bbox['x_min']), int(bbox['x_max'])
    y_min, y_max = int(bbox['y_min']), int(bbox['y_max'])

    h, w = sum_img.shape
    x_min_adj = max(0, x_min - left)
    x_max_adj = min(w, x_max - left)
    y_min_adj = max(0, y_min - top)
    y_max_adj = min(h, y_max - top)

    cropped_sum_img = sum_img[y_min_adj:y_max_adj, x_min_adj:x_max_adj]

    out_path = Path(output_path) / f"{Path(img_input_path).stem}_CROP_SUMIMG.png"
    cv2.imwrite(str(out_path), cropped_sum_img)

    return cropped_sum_img, bbox, metadata