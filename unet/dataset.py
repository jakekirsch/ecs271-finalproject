import torch
from typing import Tuple
from torch.utils.data import Dataset
import pandas as pd
import nibabel as nib
from torchvision.transforms import v2 
import time
from functools import lru_cache

# TODO: training img transforms
# TODO: binary classification on Y --> map all 1 to 1 everything else to 0

class SegThorImagesDataset(Dataset):
    """SegThor Image Dataset"""
    def __init__(self, patient_idx_file: str = 'data/train_patient_idx.csv', 
                 root_dir: str = 'data/train', transform=None, 
                 img_crop_size: int = 312,
                 mask_output_size: int = 116):
        self.file_names = pd.read_csv(patient_idx_file)
        self.root_dir = root_dir 
        self.transform = transform
        self.img_crop_size = img_crop_size
        self.mask_output_size = mask_output_size
        self.center_crop = v2.CenterCrop(312)
        self.mask_output_crop = v2.CenterCrop(self.mask_output_size)

        self.patient_x: dict[str, torch.Tensor] = {}
        self.patient_y: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.file_names)
    
    def _filenames_from_patient_name(self, patient_name: str) -> Tuple[str, str]:
        # return filenames X, Y for an idx (Patient_01.nii.gz, GT.nii.gz)
        
        patient_dir = f"{self.root_dir}/{patient_name}"
        return f"{patient_dir}/{patient_name}.nii.gz", f"{patient_dir}/GT.nii.gz"


    def _get_img_data_from_filename(self, x_file_name, y_file_name, use_cache: bool = False):
        """
        Get the image and fdata from the filenames and convert to Tensors
        """
        if x_file_name in self.patient_x.keys() and use_cache:
            print("fetching cached data")
            x_data = self.patient_x[x_file_name]
        else:
            x_img = nib.load(x_file_name)   
            x_data = x_img.get_fdata()
            x_data = torch.Tensor(x_data)
            if use_cache: 
                self.patient_x[x_file_name] = x_data
        if y_file_name in self.patient_y.keys() and use_cache:
            y_data = self.patient_y[y_file_name]
        else:    
            y_img = nib.load(y_file_name)
            y_data = y_img.get_fdata()
            y_data = torch.Tensor(y_data)
            if use_cache:
                self.patient_y[y_file_name] = y_data

        return x_data, y_data

    def __getitem__(self, idx):
        """From an idx, get filenames and slice idx from the csv file, open corresponding X,Y and return 
        the corresponding slice """
        patient_name, patient_slice = self.file_names.iloc[idx, :]
        x_file_name, y_file_name = self._filenames_from_patient_name(patient_name)
        x_data, y_data = self._get_img_data_from_filename(x_file_name, y_file_name)
        X = x_data[:, :, patient_slice].unsqueeze(0)
        Y = y_data[:, :, patient_slice]
        X = self.center_crop(X)
        Y = self.mask_output_crop(Y)
        if self.transform:
            X = self.transform(X)
        return X, Y
