"""
visulization util

Author: Muhammad Faizan
Date: 13 May 2023

"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.util import montage
from pathlib import Path
import os
import cv2
import sys
from IPython.display import Image
import imageio

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  #project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from DataLoader.dataset import BraTSDataset, get_dataloader
from config.configs import Config


def visualize_abnormal_area(image, label):
    """
    visualize the abonrmality in an image, just show the abnormal area, 
    No need to differentiate between different tumors types

    Parameters
    ----------
    image: torch.Tensor
    label: torch.Tensor
    """
    # convert to numpy array
    image_tensor = image.squeeze()[0].cpu().detach().numpy()
    label_tensor = label.squeeze()[0].cpu().detach().numpy()
    print("Num uniq Image values :", len(np.unique(image_tensor, return_counts=True)[0]))
    print("Min/Max Image values:", image_tensor.min(), image_tensor.max())
    print("Num uniq Mask values:", np.unique(label_tensor, return_counts=True))

    image = np.rot90(montage(image_tensor))
    label = np.rot90(montage(label_tensor))

    fig, axes = plt.subplots(1, 1, figsize = (20, 20))
    axes.imshow(image, cmap = 'bone')
    axes.imshow(np.ma.masked_where(label == False, label),
           cmap='cool', alpha=0.6)
    plt.title('A patient image')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
def to_categorical(mask, num_classes):
    """ 1-hot encodes a tensor 
    
    Parameters
    ----------
    mask: np.ndarray
    num_classes: int"""
    # print(f'mask data type: {mask[:, :, 78].max(), mask[:, :, 78].min()}')
    mapping = {0: 0, 1: 2, 2: 1, 4: 3}
    mapped_mask = np.vectorize(mapping.get)(mask)
    return np.eye(num_classes, dtype= np.uint8)[mapped_mask]

def channel_last(image, label):
    """convert to channel last
    
    Parameters
    ----------
    image: torch.Tensor
    label: torch.Tensor"""
    #remove batch dim
    if len(image.shape) == 5:
        image = image.squeeze(0)
        label = label.squeeze(0)
    #conver to last channel from (c, d, h, w) --> (h, w, d, c)
    image = image.permute(2, 3, 1, 0)
    image = image.numpy()
    label = label.numpy()
    return (image, label)
    

def get_labelled_image(image, label, is_categorical = False):
    """get labelled image
    
    Parameters
    ----------
    image: torch.Tensor
    label: torch.Tensor
    categorical: bool
    """
    image, label = channel_last(image, label)
    if not is_categorical:
        label = to_categorical(label.astype(np.uint8), num_classes= 4)

    image = cv2.normalize(image[:, :, :, 0], None, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
    
    labeled_image = np.zeros_like(label[:, :, :, 1:])
    #remove tumor part from image
    labeled_image[:, :, :, 0] = image * (label[:, :, :, 0])
    labeled_image[:, :, :, 1] = image * (label[:, :, :, 0])
    labeled_image[:, :, :, 2] = image * (label[:, :, :, 0])

     # color labels
    labeled_image += label[:, :, :, 1:] * 255
    return labeled_image

def visualize_data_gif(data_):
    images = []
    for i in range(data_.shape[0]):
        x = data_[min(i, data_.shape[0] - 1), :, :]
        y = data_[:, min(i, data_.shape[1] - 1), :]
        z = data_[:, :, min(i, data_.shape[2] - 1)]
        img = np.concatenate((x, y, z), axis=1)
        images.append(img)
    imageio.mimsave(f"{ROOT}/runs/gif.gif", images, duration=0.5, format='GIF')
    return Image(filename=f"{ROOT}/runs/gif.gif", format='png')
    

if __name__ == "__main__":
    json_file = Config.newGlobalConfigs.json_file
    phase = 'val'
    data = get_dataloader(BraTSDataset, 
                          Config.newGlobalConfigs.path_to_csv, 
                          phase, 1, 2, 
                          json_file=json_file,
                          is_process_mask= False)
    batch = next(iter(data))
    image, label = batch["image"], batch['label']
    print('visualizing an image with label')
    labelled_img = get_labelled_image(image, label)
    visualize_data_gif(labelled_img)
    print('Done!!')
