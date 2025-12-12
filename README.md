# VisWanda-Eco-VLM: Hardware-Aware 2:4 Pruning for Vision Transformers

> **"Smashing" Vision Models for the Edge.** > An implementation of Activation-Aware Pruning (Wanda) tailored for Vision Transformers, enforcing NVIDIA-friendly 2:4 Structured Sparsity.

## Motivation
Running huge Vision Language Models(VLMs) on the limited/consumer hardware is what every AI researcher or lab swears by, and I'm no different. Vision Transformers (ViTs) are powerful but computationally expensive, making them difficult to deploy on edge devices or in sustainable "Green AI" pipelines. Standard unstructured pruning (removing random weights) offers theoretical compression but **zero inference speedup** on modern hardware because GPUs cannot skip random zeros efficiently.

**The Solution:** 2:4 Structured Sparsity. By enforcing a pattern where exactly 2 out of every 4 consecutive weights are zero, we can unlock **2x math throughput** on NVIDIA Ampere (A100) and Hopper (H100) Tensor Cores.

## Objective
To adapt the **Wanda (Pruning by Weights and activations)** metric—originally designed for LLMs—to the **Vision Domain**, specifically targeting:

1.  **Activation-Awareness:** Preserving weights that connect to high-magnitude image patches (outliers).
2.  **Hardware-Readiness:** Enforcing 2:4 semi-structured sparsity patterns compatible with TensorRT and cuSPARSELt.
