import torch
from torch.utils.data import Dataset
import h5py

class ChessDataset(Dataset):
    def __init__(self, hdf5_path: str, chunk_size: int = 512):
        self.hdf5_path = hdf5_path
        self.chunk_size = chunk_size
        self.h5_file = None
        self.boards_dset = None
        self.policies_dset = None
        self.values_dset = None

        with h5py.File(self.hdf5_path, 'r') as f:
            self.total_samples = f['inputs'].shape[0]
            self.num_chunks = (self.total_samples + chunk_size - 1) // chunk_size 

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.chunk_size
        end = min(start + self.chunk_size, self.total_samples)

        boards = torch.from_numpy(self.boards_dset[start:end]).float()
        policies = torch.from_numpy(self.policies_dset[start:end]).float()
        values = torch.from_numpy(self.values_dset[start:end]).float()

        return boards, policies, values

    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file is not None and self.h5_file.id.valid:
            self.h5_file.close()


def _worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset_obj = getattr(worker_info.dataset, 'dataset', worker_info.dataset)

    dataset_obj.h5_file = h5py.File(dataset_obj.hdf5_path, 'r')
    dataset_obj.boards_dset = dataset_obj.h5_file['inputs']
    dataset_obj.policies_dset = dataset_obj.h5_file['policies']
    dataset_obj.values_dset = dataset_obj.h5_file['values']