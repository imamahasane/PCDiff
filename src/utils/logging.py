from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

def setup_logger(log_dir: Path, name: str = "pcdiff") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_dir / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger

def make_tb_writer(log_dir: Path) -> SummaryWriter:
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(log_dir))
