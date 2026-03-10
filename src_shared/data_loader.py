import torch
from torch.utils.data import Dataset
import h5py

class ChessDataset(Dataset):
    def __init__(self, hdf5_path: str, indices: list):
        self.hdf5_path = hdf5_path
        self.indices = indices

        # File and dataset handles start as None and are opened lazily
        self.h5_file = None
        self.boards_dset = None
        self.policies_dset = None
        self.values_dset = None

    def _open_file(self):
        """Lazy initialization of HDF5 handles for PyTorch multiprocess safety."""
        # Ensure SWMR is True if you are reading while main.py is writing
        self.h5_file = h5py.File(self.hdf5_path, 'r', swmr=True)
        
        # FIXED: Match the exact keys written by main.py
        self.boards_dset = self.h5_file['inputs'] 
        self.policies_dset = self.h5_file['policies']
        self.values_dset = self.h5_file['values']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Open the file on the very first access attempt
        if self.h5_file is None:
            self._open_file()
            
        absolute_pos = self.indices[idx]
        
        board = self.boards_dset[absolute_pos]
        policy = self.policies_dset[absolute_pos]
        value = self.values_dset[absolute_pos]

        boards_tensor = torch.from_numpy(board).float()
        policies_tensor = torch.from_numpy(policy).float()
        values_tensor = torch.tensor(value, dtype=torch.float32)

        return boards_tensor, policies_tensor, values_tensor

    def close(self):
        """Method to explicitly close the file handle."""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass
            self.h5_file = None

    def __del__(self):
        self.close()