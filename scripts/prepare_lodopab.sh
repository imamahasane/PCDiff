#!/usr/bin/env bash
set -euo pipefail

echo "LoDoPaB-CT is distributed by the dataset authors. Please download it following their official instructions,"
echo "then place the HDF5 files under: data/lodopab/"
echo ""
echo "Expected files (examples):"
echo "  data/lodopab/train.h5"
echo "  data/lodopab/val.h5"
echo "  data/lodopab/test.h5"
echo ""
echo "NOTE: src/data/lodopab.py assumes ground-truth images are released in"
echo "[0,1] (per Leuschner et al. 2021) and rescales to [-1,1] on load. It"
echo "will warn at runtime if any loaded slice falls outside [0,1] -- if you"
echo "see that warning, verify the actual value range in your downloaded"
echo "files before trusting any PSNR/SSIM numbers computed from them."
