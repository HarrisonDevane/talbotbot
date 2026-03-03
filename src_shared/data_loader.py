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
        absolute_pos = self.indices[idx]
        
        # Data fetch using worker-local handles
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

def _worker_init_fn(worker_id):
    """
    Initializes HDF5 handles in SWMR mode for each worker.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset_obj = getattr(worker_info.dataset, 'dataset', worker_info.dataset)
    
    # Open with SWMR enabled to allow concurrent reading while Orchestrator holds 'a' mode
    dataset_obj.h5_file = h5py.File(dataset_obj.hdf5_path, 'r', libver='latest', swmr=True)
    
    dataset_obj.boards_dset = dataset_obj.h5_file['inputs']
    dataset_obj.policies_dset = dataset_obj.h5_file['policies']
    dataset_obj.values_dset = dataset_obj.h5_file['values']