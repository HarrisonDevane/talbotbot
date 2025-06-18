import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import h5py 
import time

class ChessDataset(Dataset):
    # ... (rest of ChessDataset class remains exactly the same as last update) ...
    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        self.h5_file = None 
        self.boards_dset = None
        self.policies_dset = None
        self.values_dset = None

        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found at: {self.hdf5_path}")
        
        try:
            with h5py.File(self.hdf5_path, 'r') as temp_h5_file:
                self.num_samples = temp_h5_file['inputs'].shape[0]
            print(f"Dataset initialized (main process): Found {self.num_samples} samples in {self.hdf5_path}")
        except Exception as e:
            raise IOError(f"Could not open HDF5 file at {self.hdf5_path} to determine length: {e}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.h5_file is None:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info else 'main'
            print(f"WARNING: HDF5 file not opened for {worker_id}. Opening now (consider worker_init_fn).")
            self.h5_file = h5py.File(self.hdf5_path, 'r')
            self.boards_dset = self.h5_file['inputs']
            self.policies_dset = self.h5_file['policies']
            self.values_dset = self.h5_file['values']

        # start_total = time.perf_counter() 
        
        # hdf5_read_start = time.perf_counter()
        board_np = self.boards_dset[idx]
        policy_np = self.policies_dset[idx]
        value_np = self.values_dset[idx]
        # hdf5_read_end = time.perf_counter()

        # board_to_tensor_start = time.perf_counter()
        board_tensor = torch.from_numpy(board_np).float() 
        # board_to_tensor_end = time.perf_counter()

        # policy_to_tensor_start = time.perf_counter()
        policy_index = torch.tensor(policy_np, dtype=torch.long)
        # policy_to_tensor_end = time.perf_counter()

        # value_to_tensor_start = time.perf_counter()
        value_target = torch.tensor(value_np, dtype=torch.float32) 
        # value_to_tensor_end = time.perf_counter()
        
        # total_get_item_time_ms = (time.perf_counter() - start_total) * 1000

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0 
        # worker_id_str_for_log = f"Worker {worker_id}"

        # if idx % 1000 == 0: 
        #      print(
        #         f"{worker_id_str_for_log} - Sample {idx}: "
        #         f"HDF5 Read: {(hdf5_read_end - hdf5_read_start)*1000:.2f}ms, "
        #         f"Board to Tensor: {(board_to_tensor_end - board_to_tensor_start)*1000:.2f}ms, "
        #         f"Policy to Tensor: {(policy_to_tensor_end - policy_to_tensor_start)*1000:.2f}ms, "
        #         f"Value to Tensor: {(value_to_tensor_end - value_to_tensor_start)*1000:.2f}ms, "
        #         f"Total __getitem__: {total_get_item_time_ms:.2f}ms"
        #     )

        return board_tensor, policy_index, value_target

    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            if self.h5_file.id.valid: 
                print(f"Closing HDF5 file: {self.h5_file.filename}")
                self.h5_file.close()

# This function will be passed to DataLoader as worker_init_fn
def _worker_init_fn(worker_id):
    """
    Initializes each DataLoader worker by opening its own HDF5 file handle.
    This prevents h5py file objects from being pickled.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        # Access the original dataset through the 'dataset' attribute of the Subset object
        # The .dataset attribute of a Subset object points to the underlying original dataset.
        dataset_obj = worker_info.dataset.dataset 
        
        # Open the HDF5 file for this specific worker
        print(f"Worker {worker_id}: Opening HDF5 file {dataset_obj.hdf5_path}")
        dataset_obj.h5_file = h5py.File(dataset_obj.hdf5_path, 'r')
        dataset_obj.boards_dset = dataset_obj.h5_file['inputs']
        dataset_obj.policies_dset = dataset_obj.h5_file['policies']
        dataset_obj.values_dset = dataset_obj.h5_file['values']