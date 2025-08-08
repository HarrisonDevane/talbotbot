import torch
from torch.utils.data import Dataset
import numpy as np
import os
import h5py
import logging
import time # Added for profiling

class ChessDataset(Dataset):
    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        self.h5_file = None
        
        with h5py.File(self.hdf5_path, 'r') as h5_file:
            self.boards = torch.from_numpy(h5_file['inputs'][...]).float()
            self.policies = torch.from_numpy(h5_file['policies'][...]).float()
            self.values = torch.from_numpy(h5_file['values'][...]).float()

        self.num_samples = len(self.boards)


    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # This is now a simple, lightning-fast in-memory lookup
        return self.boards[idx], self.policies[idx], self.values[idx]
    

    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            if self.h5_file.id.valid:
                # Use logger to indicate file closure
                self.logger.info(f"Closing HDF5 file: {self.h5_file.filename}")
                self.h5_file.close()

# This function will be passed to DataLoader as worker_init_fn
def _worker_init_fn(worker_id):
    """
    Initializes each DataLoader worker by opening its own HDF5 file handle.
    This prevents h5py file objects from being pickled.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        # Access the Subset object
        subset_dataset = worker_info.dataset
        # Access the original ChessDataset object via the .dataset attribute
        original_dataset = subset_dataset.dataset
        
        # Now, use the original_dataset object to access the logger and file attributes
        # No need for the file existence check here, as it's done in the main thread
        original_dataset.h5_file = h5py.File(original_dataset.hdf5_path, 'r')
        original_dataset.boards_dset = original_dataset.h5_file['inputs']
        original_dataset.policies_dset = original_dataset.h5_file['policies']
        original_dataset.values_dset = original_dataset.h5_file['values']