import os
import random
from typing import Optional

import numpy as np
import torch

def seed_everything(seed: int, deterministic: bool = True) -> None:
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Deterministic behavior where possible.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For matmul determinism (may reduce performance)
        torch.use_deterministic_algorithms(False)

def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
