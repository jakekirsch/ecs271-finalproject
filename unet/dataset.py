import torch
from typing import Tuple
from torch.utils.data import Dataset
import pandas as pd
import nibabel as nib
# need csv of patient file names 

class SegThorImagesDataset(Dataset):
    """SegThor Image Dataset"""
    def __init__(self, patient_idx_file: str = 'data/train_patient_idx.csv', 
                 root_dir: str = 'data/train', transform=None):
        self.file_names = pd.read_csv(patient_idx_file)
        self.root_dir = root_dir 
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_names)
    
    def _filenames_from_patient_name(self, patient_name: str) -> Tuple[str, str]:
        # return filenames X, Y for an idx (Patient_01.nii.gz, GT.nii.gz)
        
        patient_dir = f"{self.root_dir}/{patient_name}"
        return f"{patient_dir}/{patient_name}.nii.gz", f"{patient_dir}/GT.nii.gz"


    def __getitem__(self, idx):
        """From an idx, get filenames and slice idx from the csv file, open corresponding X,Y and return 
        the corresponding slice """
        patient_name, patient_slice = self.file_names.iloc[idx, :]
        x_file_name, y_file_name = self._filenames_from_patient_name(patient_name)
        print(f"Opening: {x_file_name}")
        x_img = nib.load(x_file_name)
        y_img = nib.load(y_file_name)
        x_data = x_img.get_fdata()
        x_data = torch.Tensor(x_data)
        y_data = y_img.get_fdata()
        y_data = torch.Tensor(y_data)

        print(f"Returning slice: {patient_slice}")
        X = x_data[:, :, patient_slice].unsqueeze(0)
        Y = y_data[:, :, patient_slice]

        return X, Y
