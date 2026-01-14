# Physics-Conditioned Diffusion Model for Low-Dose CT Reconstruction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official implementation of the paper:

> **“Physics-Conditioned Diffusion Model for Low-Dose CT Reconstruction”**  
> *Md Imam Ahasan, Guangchao Yang, A. F. M. Abdun Noor, Mohammad Azam Khan*  
> Submitted to **28th International Conference on Pattern Recognition (2026)**, January 2026.

---

## 🔍 Overview

Low-dose CT (LDCT) reconstruction is a severely ill-posed inverse problem due to quantum noise and incomplete measurements.  
This work proposes a **physics-conditioned diffusion model** that explicitly incorporates the CT forward model into the diffusion process.

The model leverages:
- Persistent conditioning using the **adjoint backprojection**
- A **physics-consistency loss** enforced in the sinogram domain
- Conditional diffusion sampling for **uncertainty estimation**

---

## ✨ Key Contributions

- Physics-informed conditional diffusion framework for LDCT
- Persistent conditioning with adjoint operator \( A^*(y) \)
- Joint optimization of diffusion loss and physics-consistency loss
- End-to-end differentiable CT forward and adjoint operators
- Uncertainty estimation via multiple conditional diffusion samples
- Fully reproducible, research-grade PyTorch codebase

---

## 🧠 Method Summary

### Conditioning Mechanism

At each diffusion timestep \( t \), the denoising network is conditioned on:
- Noisy image \( x_t \)
- Physics prior \( c = A^*(y) \)
- Diffusion timestep embedding \( t \)

The conditioning signal is **persistently applied at every diffusion step**.

---

### Training Objective

The total loss function is defined as:
\[
\mathcal{L} =
\underbrace{\|\epsilon - \epsilon_\theta(x_t, c, t)\|^2}_{\text{Diffusion Loss}}
+
\lambda
\underbrace{\|A(\hat{x}_0) - y\|^2}_{\text{Physics Consistency Loss}}
\]

---

### Uncertainty Estimation

Uncertainty is estimated by:
1. Drawing multiple conditional diffusion samples for the same measurement
2. Computing pixel-wise variance or standard deviation across samples

---

## 📁 Repository Structure

