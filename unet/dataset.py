import torch
from typing import Tuple
from torch.utils.data import Dataset
import pandas as pd
import nibabel as nib
from torchvision.transforms import v2 
from torchvision import tv_tensors
import time
from functools import lru_cache

# TODO: training img transforms
# TODO: binary classification on Y --> map all 1 to 1 everything else to 0
from collections import OrderedDict
from typing import Tuple

class LRUCache:
    """A simple LRU (Least Recently Used) Cache implemented with OrderedDict."""
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str):
        if key in self.cache:
            self.cache.move_to_end(key)  # mark as most recently used
            return self.cache[key]
        return None

    def put(self, key: str, value: torch.Tensor):
        if key in self.cache:
            self.cache.move_to_end(key)  # mark as most recently used
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # remove least recently used item


class SegThorImagesDataset(Dataset):
    """SegThor Image Dataset"""
    def __init__(self, patient_idx_file: str = 'data/train_patient_idx.csv', 
                 root_dir: str = 'data/train', transform=None, 
                 img_crop_size: int = 312,
                 mask_output_size: int = 116,
                 cache_size: int = 1):
        self.file_names = pd.read_csv(patient_idx_file)
        self.root_dir = root_dir 
        self.transform = transform
        self.img_crop_size = img_crop_size
        self.mask_output_size = mask_output_size
        self.center_crop = v2.CenterCrop(312)
        self.mask_output_crop = v2.CenterCrop(self.mask_output_size)

        self.patient_x_cache = LRUCache(cache_size)
        self.patient_y_cache = LRUCache(cache_size)

    def __len__(self) -> int:
        return len(self.file_names)
    
    def _filenames_from_patient_name(self, patient_name: str) -> Tuple[str, str]:
        # return filenames X, Y for an idx (Patient_01.nii.gz, GT.nii.gz)
        
        patient_dir = f"{self.root_dir}/{patient_name}"
        return f"{patient_dir}/{patient_name}.nii.gz", f"{patient_dir}/GT.nii.gz"


    def _get_img_data_from_filename(self, x_file_name, y_file_name):
        """
        Get the image and data from the filenames and convert to Tensors
        """
        x_data = self.patient_x_cache.get(x_file_name)
        if x_data is None:
            x_img = nib.load(x_file_name)
            x_data = x_img.get_fdata()
            x_data = torch.Tensor(x_data)
            self.patient_x_cache.put(x_file_name, x_data)
        
        y_data = self.patient_y_cache.get(y_file_name)
        if y_data is None:
            y_img = nib.load(y_file_name)
            y_data = y_img.get_fdata()
            y_data = torch.Tensor(y_data)
            self.patient_y_cache.put(y_file_name, y_data)

        return x_data, y_data
    
    def __getitem__(self, idx):
        """From an idx, get filenames and slice idx from the csv file, open corresponding X,Y and return 
        the corresponding slice """
        patient_name, patient_slice = self.file_names.iloc[idx, :]
        x_file_name, y_file_name = self._filenames_from_patient_name(patient_name)
        x_data, y_data = self._get_img_data_from_filename(x_file_name, y_file_name)
        X = x_data[:, :, patient_slice].unsqueeze(0)
        Y = y_data[:, :, patient_slice]
        Y_m = tv_tensors.Mask(Y)
        if self.transform:
            X, Y_m = self.transform(X, Y_m)
        X = self.center_crop(X)
        Y_m = self.mask_output_crop(Y_m)

        return X, Y_m
