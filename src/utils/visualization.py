import cv2

from matplotlib.pyplot import plt
import matplotlib.patches as patches

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