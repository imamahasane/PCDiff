from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

class MayoAAPMDataset(Dataset):


    def __init__(self, root: str | Path, split: str):
        super().__init__()
        self.root = Path(root) / split
        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"No .npz files found in {self.root}. Run scripts/prepare_mayo_aapm.py first."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        d = np.load(self.files[idx])
        x = torch.from_numpy(d["x"]).float()  # (1,H,W)
        y = torch.from_numpy(d["y"]).float()  # (angles,det)
        return {"x": x, "y": y}
