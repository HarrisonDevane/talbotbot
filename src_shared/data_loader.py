import torch
from torch.utils.data import Dataset
import h5py

class ChessDataset(Dataset):
    def __init__(self, hdf5_path: str, indices: list):
        self.hdf5_path = hdf5_path
        self.indices = indices

        # File and dataset handles will be opened by _worker_init_fn
        self.h5_file = None
        self.boards_dset = None
        self.policies_dset = None
        self.values_dset = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
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
        if hasattr(self, 'h5_file') and self.h5_file is not None and self.h5_file.id.valid:
            self.h5_file.close()
            self.h5_file = None

    def __del__(self):
        self.close()


def _worker_init_fn(worker_id):
    """
    Initializes the HDF5 file handle for each DataLoader worker. 
    This is necessary to safely use h5py with multi-process loading.
    """
    worker_info = torch.utils.data.get_worker_info()
    
    dataset_obj = getattr(worker_info.dataset, 'dataset', worker_info.dataset)
    dataset_obj.h5_file = h5py.File(dataset_obj.hdf5_path, 'r')
    
    # Assign the dataset handles
    dataset_obj.boards_dset = dataset_obj.h5_file['inputs']
    dataset_obj.policies_dset = dataset_obj.h5_file['policies']
    dataset_obj.values_dset = dataset_obj.h5_file['values']