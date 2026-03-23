import torch
import random
from torch.utils.data import Dataset
import h5py
import src_shared.utils as utils

class ChessDataset(Dataset):
    def __init__(self, hdf5_path: str, indices: list, augment: bool):
        self.hdf5_path = hdf5_path
        self.indices = indices
        self.augment = augment

        # File and dataset handles start as None and are opened lazily
        self.h5_file = None
        self.boards_dset = None
        self.policies_dset = None
        self.values_dset = None
        self.masks_dset = None

    def _open_file(self):
        """Lazy initialization of HDF5 handles for PyTorch multiprocess safety."""
        self.h5_file = h5py.File(self.hdf5_path, 'r', swmr=True)
        
        self.boards_dset = self.h5_file['inputs'] 
        self.policies_dset = self.h5_file['policies']
        self.values_dset = self.h5_file['values']
        self.masks_dset = self.h5_file['legal_masks']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self._open_file()
            
        absolute_pos = self.indices[idx]
        
        board = self.boards_dset[absolute_pos]
        policy = self.policies_dset[absolute_pos]
        value = self.values_dset[absolute_pos]
        mask = self.masks_dset[absolute_pos]

        boards_tensor = torch.from_numpy(board).float()
        policies_tensor = torch.from_numpy(policy).float()
        values_tensor = torch.tensor(value, dtype=torch.float32)
        masks_tensor = torch.from_numpy(mask).bool()

        if self.augment:
            # 1. Horizontal Flip (2x) - Valid if no castling
            if boards_tensor[13:17, :, :].sum() == 0.0 and random.random() > 0.5:
                boards_tensor, policies_tensor, masks_tensor = utils.apply_horizontal_flip_torch(
                    boards_tensor, policies_tensor, masks_tensor
                )

            # 2. Vertical Flip (Adds 2x -> Total 4x)
            # Valid ONLY if no castling AND no pawns exist in current or historical plies
            # Pawn planes: 0, 6 (current) and 18, 24, 30, 36, 42, 48, 54, 60 (history)
            pawn_indices = [0, 6, 18, 24, 30, 36, 42, 48, 54, 60]
            
            if boards_tensor[pawn_indices, :, :].sum() == 0.0 and random.random() > 0.5:
                boards_tensor, policies_tensor, masks_tensor = utils.apply_vertical_flip_torch(
                    boards_tensor, policies_tensor, masks_tensor
                )

        return boards_tensor, policies_tensor, values_tensor, masks_tensor

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