from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class LoDoPaBDataset(Dataset):


    def __init__(self, root: str | Path, split: str):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.files = sorted(glob.glob(str(self.root / f"{split}*.h5")))
        if not self.files:
            raise FileNotFoundError(
                f"No HDF5 files found for split='{split}' in {self.root}.\n"
                "Place LoDoPaB h5 files under data/lodopab/, e.g., train.h5 / validation.h5 / test.h5."
            )

        # Open all files lazily in __getitem__ (h5py not fork-safe).
        self.index = []  # list of (file_path, local_idx)
        for fp in self.files:
            with h5py.File(fp, "r") as f:
                images_key, sino_key = self._infer_keys(f)
                n = f[images_key].shape[0]
            for i in range(n):
                self.index.append((fp, i))

    def _infer_keys(self, f: h5py.File) -> Tuple[str, str]:
        # Common LoDoPaB keys
        candidates = [
            ("images", "sinograms"),
            ("image", "sinogram"),
            ("ground_truth", "observation"),
            ("x", "y"),
        ]
        for ik, sk in candidates:
            if ik in f and sk in f:
                return ik, sk
        # Try nested groups
        for g in f.keys():
            if isinstance(f[g], h5py.Group):
                grp = f[g]
                for ik, sk in candidates:
                    if ik in grp and sk in grp:
                        return f"{g}/{ik}", f"{g}/{sk}"
        raise KeyError(f"Could not infer dataset keys in file. Keys: {list(f.keys())}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fp, j = self.index[idx]
        with h5py.File(fp, "r") as f:
            images_key, sino_key = self._infer_keys(f)
            x = f[images_key][j]  # (H,W)
            y = f[sino_key][j]    # (angles,det)
        x = torch.from_numpy(np.asarray(x)).float().unsqueeze(0)  # (1,H,W)
        y = torch.from_numpy(np.asarray(y)).float()               # (angles,det)
        return {"x": x, "y": y}
