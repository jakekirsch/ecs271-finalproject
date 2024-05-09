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

import nibabel as nib

def load_dataset(image, label):
    file1 = nib.load(image)
    file2 = nib.load(label)
    data1 = file1.get_fdata()
    data2 = file2.get_fdata()
    print(data1.shape)
    return data1, data2


# for each file in the directory
def unpack_images(root: str = 'data/train', output_dir: str = 'data/processed/train'):
    # Check if the directory exists
    if not os.path.exists(output_dir):
        # If it doesn't exist, create it
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created successfully.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    for patient_dir in os.listdir(root):
        if os.path.isdir(os.path.join(root, patient_dir)):
            # get the image data as Tensor
            print(f"Opening: {patient_dir}")
            img = nib.load(os.path.join(root, f"{patient_dir}/{patient_dir}.nii.gz"))
            gt_img = nib.load(os.path.join(root, f'{patient_dir}/GT.nii.gz'))
            img_data = img.get_fdata()
            gt_data = img.get_fdata()
            img_data = torch.Tensor(img_data)
            gt_data = torch.Tensor(gt_data)
            # slices is first index 
            num_slices = img_data.size()[0]
            for idx in range(num_slices):
                slice_to_save = img_data[:, :, 1].unsqueeze(0)
                gt_to_save = gt_data[:, :, 1]

                slice_filename = f"{output_dir}/{patient_dir}_{idx}_X.pt"
                gt_filename = f"{output_dir}/{patient_dir}_{idx}_Y.pt"
                if not os.path.exists(slice_filename):
                    torch.save(slice_to_save, slice_filename)
                if not os.path.exists(gt_filename):
                    torch.save(gt_to_save, gt_filename)
