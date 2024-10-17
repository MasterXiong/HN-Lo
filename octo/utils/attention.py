import cv2
import numpy as np
import matplotlib.cm as cm


def generate_attention_map(patch_attention, image_size=(256, 256), patch_size=(16, 16)):
    heatmap = np.zeros(image_size)
    max_row = image_size[0] // patch_size[0]
    max_col = image_size[1] // patch_size[1]
    for i, w in enumerate(patch_attention):
        row = i // max_col
        col = i % max_col
        heatmap[row * patch_size[0]:(row + 1) * patch_size[0], col * patch_size[1]:(col + 1) * patch_size[1]] = w
    return heatmap


def combine_image_and_heatmap(image, heatmap):
    # Ensure the heatmap is between 0 and 1
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
    # Convert the heatmap to a color map (using 'jet' colormap here)
    heatmap_colored = cm.jet(heatmap_normalized)[:, :, :3]  # Remove the alpha channel
    # Convert the heatmap from matplotlib format to the format needed by OpenCV
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    # Resize heatmap to match the original image size
    heatmap_resized = cv2.resize(heatmap_colored, (image.shape[0], image.shape[1]))
    # Overlay the heatmap onto the original image
    overlay = cv2.addWeighted(image, 0.6, heatmap_resized, 0.4, 0)
    return overlay
