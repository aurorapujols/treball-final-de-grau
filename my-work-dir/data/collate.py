import torch
import torch.nn.functional as F

def pad_collate(batch):
    """
    batch: list of (img, filename)
    img shape: (1, H, W)
    """
    images, filenames = zip(*batch) # ((img1, fname1), (img2, fname2)) -> ((img1, img2), (fname1, fname2))

    # Find max height and width in this batch
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    # Make each image as big as the max width and height in the batch adding black pixels in the right and bottom
    padded_images = []
    for img in images:
        _, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w

        # Compute symmetrix padding
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # Pad: (left, right, top, bottom)
        padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom))
        padded_images.append(padded)

    # stack images of size (C, max_h, max_w) into a tensor of size (batch_size, C, max_h, max_w)
    return torch.stack(padded_images), filenames