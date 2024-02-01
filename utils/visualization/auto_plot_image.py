import torch
import numpy as np
import cv2

def plot_image(save_path, image, convert_RGB2BGR=True):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = image.astype(float)
    if image.max() < 1.1 and image.min() > -0.1: # [0, 1]
        image = image * 255
    elif image.max() < 1.1 and image.min() > -1.1: # [-1, 1]
        image = (image + 1.0) * 0.5 * 255
    image = image.clip(0, 255)  
    image = image.astype(np.uint8)
    if len(image.shape) == 4 and image.shape[0] == 1:
        image = image[0]
    if len(image.shape) == 3 and image.shape[0] <= 4: # C, H, W
        image = torch.from_numpy(image).permute(1, 2, 0).numpy()
    if len(image.shape) == 3 and convert_RGB2BGR:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image)