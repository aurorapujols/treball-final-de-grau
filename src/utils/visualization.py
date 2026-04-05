import os
import cv2
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image, ImageEnhance

def print_bounding_box(img_bgr, bbox):

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Get size of bounding box
    width = bbox['x_max'] - bbox['x_min']
    height = bbox['y_max'] - bbox['y_min']

    # Plot image with bounding box
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    bounding_box = patches.Rectangle((bbox['x_min'], bbox['y_min']), width, height, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(bounding_box)
    plt.title("Image with bounding box")
    plt.axis('off')
    plt.show()

def get_average_images(df, label, n_needed, images_folder, suffix_length):
    images = []
    all_class_filenames = list(df[df['class'] == label]['filename'])
    selected_files = random.sample(all_class_filenames, n_needed)
    print(f"Selected files from class {label} to extract: {selected_files}")
    for file in os.listdir(images_folder):
        basename = os.path.splitext(os.path.basename(file))[0][:-suffix_length]
        if basename in selected_files:
            images.append(Image.open(f"{images_folder}/{file}"))
    return images

def average_image(images, resize=False):
    processed = []

    for img in images:
        if resize:
            img = img.resize((255, 255))
        processed.append(np.array(img).astype(np.float32))

    stack = np.stack(processed, axis=0)
    avg = np.mean(stack, axis=0)

    avg_img = Image.fromarray(avg.astype(np.uint8))
    enhancer = ImageEnhance.Contrast(avg_img)
    return enhancer.enhance(10.0)

def print_average_pixels(config):

    dataset_path = f"{config.paths.datasets}/{config.dataset.labeling_results}"
    labeled_df = pd.read_csv(dataset_path, sep=";")

    counts = dict(labeled_df['class'].value_counts())
    for k,v in counts.items():
        counts[k] = int(v)
    print(f"Counts: {counts}")

    for label, count in counts.items():

        df = labeled_df
        n = 0

        if label == "meteor":
            n = 200
            df = pd.read_csv(f"{config.paths.datasets}/{config.dataset.not_trained}", sep=";")
        else:
            n = min(count, 200) #min(v, 10)

        images = get_average_images(df, label, n, config.paths.processed.sum_image, suffix_length=7)
        images_crop = get_average_images(df, label, n, config.paths.processed.original, suffix_length=12)
        avg_image = average_image(images)
        avg_image_crop = average_image(images_crop, resize=True)
        avg_image.save(f"logs/images/{label}_avg{n}.png")
        avg_image_crop.save(f"logs/images/{label}_avg{n}_crop.png")

def compute_kpis(img):
    arr = np.asarray(img, dtype=np.float32)

    mean_val = arr.mean()
    return {
        "mean": mean_val,
        "median": np.median(arr),
        "p5": np.percentile(arr, 5),
        "p95": np.percentile(arr, 95),
        "dynamic_range": arr.max() - arr.min(),
        "contrast": arr.std() / (mean_val + 1e-6)
    }

def compute_dataset_kpis(df, images_folder, suffix):
    rows = []

    filenames = df["filename"].values
    labels = df["class"].values
    total = len(df)

    print(f"\nProcessing {total} images for suffix '{suffix}'")

    for i, (filename, label) in enumerate(zip(filenames, labels), 1):

        path = f"{images_folder}/{filename}_{suffix}.png"

        try:
            with Image.open(path) as img:
                img = img.convert("L")        # grayscale = faster
                img = img.resize((256, 256))  # huge speedup
                kpis = compute_kpis(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        kpis["filename"] = filename
        kpis["class"] = label
        rows.append(kpis)

        # simple progress indicator
        if i % 1000 == 0:
            print(f"  {i}/{total} images processed...")

    print(f"Finished {suffix} ({len(rows)} valid images)\n")
    return pd.DataFrame(rows)

def compute_overall_kpis(config):

    dataset_path = f"{config.paths.datasets}/{config.dataset.not_trained}"
    df = pd.read_csv(dataset_path, sep=";")

    df_kpis_sumimg = compute_dataset_kpis(
        df, 
        config.paths.processed.sum_image, 
        suffix="SUMIMG"
    )

    df_kpis_crops = compute_dataset_kpis(
        df, 
        config.paths.processed.original, 
        suffix="CROP_SUMIMG"
    )

    class_summary_sumimg = df_kpis_sumimg.groupby("class").mean(numeric_only=True)
    class_summary_crops = df_kpis_crops.groupby("class").mean(numeric_only=True)

    print("=== SUMIMG KPIs ===")
    print(class_summary_sumimg)

    print("\n=== CROP KPIs ===")
    print(class_summary_crops)

    return class_summary_sumimg, class_summary_crops