import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class ChessDataset(Dataset):
    def __init__(self, hdf5_path: str, indices: list):
        """
        Standard Dataset for non-persistent training steps.
        :param indices: The specific absolute positions to sample for this batch.
        """
        self.hdf5_path = hdf5_path
        self.indices = indices

        # Handles initialized per-worker by _worker_init_fn
        self.h5_file = None
        self.boards_dset = None
        self.policies_dset = None
        self.values_dset = None


    def __len__(self):
        return len(self.indices)


    def __getitem__(self, idx):
        """
        Fetches specific data at the requested index.
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')
            self.boards_dset = self.h5_file['inputs']
            self.policies_dset = self.h5_file['policies']
            self.values_dset = self.h5_file['values']

        absolute_pos = self.indices[idx]
        
        board = self.boards_dset[absolute_pos]
        policy = self.policies_dset[absolute_pos]
        value = self.values_dset[absolute_pos]

        return torch.from_numpy(board).float(), \
               torch.from_numpy(policy).float(), \
               torch.tensor(value, dtype=torch.float32)
    

    def close(self):
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            try:
                if self.h5_file.id.valid:
                    self.h5_file.close()
            except:
                pass
            self.h5_file = None


    def __del__(self):
        self.close()