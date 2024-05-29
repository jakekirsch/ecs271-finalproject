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
    print(f"Opening: {example_filename}")
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


def plot_XY(data, gt_data, titles = ('Training Data', 'Ground Truth')):
    # plot ground truth and data slices side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
    axes[0].imshow(data, interpolation='nearest')
    axes[0].set_title(titles[0])
    axes[1].imshow(gt_data, interpolation='nearest')
    axes[1].set_title(titles[1])
    plt.show()    
    
def plot_XY_pred_class(model, X, Y):
    # plot the predicted classes into a single chart
    pred_classes = model.predict_classes(X.unsqueeze(0))
    pred_classes = pred_classes.to('cpu')
    X = X.to('cpu')
    Y = Y.to('cpu')
    plot_XY(X[0, :, :].squeeze(0), Y.squeeze(0))
    plot_XY(X[0, :, :].squeeze(0), pred_classes.squeeze(0), titles=("training data", "predicted classes"))
        

def plot_XY_for_preds(model, X, Y):
    # plot the predicted probabilities for each class 
    probas = model.predict_probabilities(X.unsqueeze(0))
    pred = probas.detach()
    plot_XY(X[0, :, :].squeeze(0), Y.squeeze(0))
    
    for i in range(4):
        plot_XY(X[0, :, :].squeeze(0), pred.squeeze(0)[i,:,:],
                                titles=("Training Data", f"Predictions Class {i}"))        
        
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


from collections import defaultdict
import pandas as pd 
# generate an index csv file 
def gen_index_file(root: str = 'data/train', overwrite: bool = False, test_patients = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]):
    """
    Given a root directory containing the .nii images, generate a corresponding index file 
    that can be used for Dataset 
    """
    # check if file exists 
    train_filename = 'data/train_patient_idx.csv'
    test_filename = 'data/test_patient_idx.csv'
    if os.path.exists(train_filename) and overwrite is False:
        print(f"Filename: {train_filename} already exists, skipping gen")
        return
    
    # check if 
    if os.path.exists(test_filename) and overwrite is False:
        print(f"Filename: {test_filename} already exists, skipping")
        return 
    
    
    # Convert list of test patients to a list of strings
    test_patient_strs = [str(patient) for patient in test_patients]
    # Function to check if directory name contains any of the test patient digits
    def contains_test_patient(directory_name, patient_list):
        for patient in patient_list:
            if patient in directory_name:
                return True
        return False

    patients = defaultdict(list)
    train_index = []
    test_index = []
    for patient_dir in os.listdir(root):
        if os.path.isdir(os.path.join(root, patient_dir)):
            # get the image data as Tensor
            img = nib.load(os.path.join(root, f"{patient_dir}/{patient_dir}.nii.gz"))
            img_data = img.get_fdata()
            img_data = torch.Tensor(img_data)
            # slices is last index
            num_slices = img_data.size()[2]
            # turn into a list of tuples, [(patient_01, 1),...]
            patient = [patient_dir]*num_slices 
            patient_index = [(patient, idx) for patient, idx in zip(patient, range(num_slices))]
            patients[patient_dir] = patient_index
            if contains_test_patient(patient_dir, test_patient_strs):
                test_index.extend(patient_index)
            else:
                train_index.extend(patient_index)
    
    test_index.sort()
    train_index.sort()
    
    train_df = pd.DataFrame(train_index, columns=['patient', 'slice_idx'])
    test_df = pd.DataFrame(test_index, columns=['patient', 'slice_idx'])

    train_df.to_csv(train_filename, index=False)
    test_df.to_csv(test_filename, index=False)
    return train_filename, test_filename

