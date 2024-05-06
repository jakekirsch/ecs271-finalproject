import torch
import os
import numpy as np
from nibabel.testing import data_path 
from matplotlib import pyplot as plt
import nibabel as nib


def zero_pad(num: int) -> str:
    if num < 10:
        return f'0{num}'
    else:
        return str(num)

def get_sample_data(dir: str = 'train', patient_number: int = 1, slice: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    root = 'data'
    patient_stub = 'Patient_' + zero_pad(patient_number)
    example_filename = os.path.join(root, f'{dir}/{patient_stub}/{patient_stub}.nii.gz')
    img = nib.load(example_filename)
    # convert to Numpy array, can then convert to Tensor for pytorch model training
    data = img.get_fdata()
    
    gt = os.path.join(root, f'{dir}/{patient_stub}/GT.nii.gz')
    gt_img = nib.load(gt)
    gt_data = gt_img.get_fdata()

    return torch.Tensor(data), torch.Tensor(gt_data)
    # convert to Tensor


def plot_slice(data, gt_data, slice_idx):
    # plot ground truth and data slices side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
    axes[0].imshow(data[:, :, slice_idx], interpolation='nearest')
    axes[0].set_title('Training Data')
    axes[1].imshow(gt_data[:, :, slice_idx], interpolation='nearest')
    axes[1].set_title('Ground Truth')
    plt.show()
