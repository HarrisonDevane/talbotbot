import torch
from torch.utils.data import Dataset
import numpy as np
import src_shared.utils as u

class ChessDataset(Dataset):
    def __init__(self, buffer_dir: str, indices: list, max_capacity: int):
        self.buffer_dir = buffer_dir
        self.indices = indices
        self.max_capacity = max_capacity
        
        # We don't load the files in __init__ to keep it multiprocessing-safe.
        self.inputs = None
        self.policies = None
        self.values = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.inputs is None:
            import os
            # Mode 'c' (copy-on-write) or 'r' ensures we don't hold a write lock
            self.inputs = np.memmap(os.path.join(self.buffer_dir, 'inputs.bin'), dtype='float16', mode='r', shape=(self.max_capacity, u.INPUT_CHANNELS, u.BOARD_DIM, u.BOARD_DIM))
            self.policies = np.memmap(os.path.join(self.buffer_dir, 'policies.bin'), dtype='float16', mode='r', shape=(self.max_capacity, u.TOTAL_POLICY_MOVES))
            self.values = np.memmap(os.path.join(self.buffer_dir, 'values.bin'), dtype='float16', mode='r', shape=(self.max_capacity,))
            
        absolute_pos = self.indices[idx]
        
        board = np.copy(self.inputs[absolute_pos])
        policy = np.copy(self.policies[absolute_pos])
        value = np.copy(self.values[absolute_pos])

        return torch.from_numpy(board).float(), \
               torch.from_numpy(policy).float(), \
               torch.tensor(value, dtype=torch.float32)