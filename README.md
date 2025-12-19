# VisWanda-Eco-VLM: Hardware-Aware 2:4 Pruning for Vision Transformers

> **"Smashing" Vision Models for the Edge.** An implementation of Activation-Aware Pruning (Wanda) tailored for Vision Transformers, enforcing NVIDIA-friendly 2:4 Structured Sparsity.

## 📸 Find the Live App here: [pruned-ViT](https://huggingface.co/spaces/Tekraj15/VisWanda-Eco-VLM)

## Motivation
Running huge Vision Language Models(VLMs) on the limited/consumer hardware is what every AI researcher or lab swears by, and I'm no different. Vision Transformers (ViTs) are powerful but computationally expensive, making them difficult to deploy on edge devices or in sustainable "Green AI" pipelines. Standard unstructured pruning (removing random weights) offers theoretical compression but **zero inference speedup** on modern hardware because GPUs cannot skip random zeros efficiently.

**The Solution:** 2:4 Structured Sparsity. By enforcing a pattern where exactly 2 out of every 4 consecutive weights are zero, we can unlock **2x math throughput** on NVIDIA Ampere (A100) and Hopper (H100) Tensor Cores.

## Objective
To adapt the **Wanda (Pruning by Weights and activations)** metric, which is originally designed for LLMs, to the **Vision Domain**, specifically targeting:

1.  **Activation-Awareness:** Preserving weights that connect to high-magnitude image patches (outliers).
2.  **Hardware-Readiness:** Enforcing 2:4 semi-structured sparsity patterns compatible with TensorRT and cuSPARSELt.

## Methodology

> **Logical workflow**: Data $\to$ Math $\to$ Masking $\to$ Validation.

Our approach adapts the Wanda (Weight and Activation) pruning metric to the vision domain, enforcing 2:4 Structured Sparsity to ensure real-world inference acceleration on NVIDIA Ampere/Hopper GPUs.





## i. Data Calibration (The "Wanda" Process)
To compress the model without retraining, we perform One-Shot Calibration:
1. Data Ingestion: We stream 128 random samples from the ImageNet-1k training set.

2. Activation Hooking: We pass these 128 images through the unpruned Vision Transformer.

3. Feature Norm Calculation: For every Linear layer in the model, we record the input activation norm $||X||_2$ averaged across the 128 samples.

Why? This identifies "salient" features. If a specific channel (e.g., Channel 45 in Layer 3) consistently has high values for these 128 images, it likely encodes important visual information (like "edges" or "eyes").

4. Metric Computation: We calculate the pruning score $S = |W| \cdot ||X||$.

5.Mask Generation: We generate a binary mask $M$ enforcing the 2:4 sparsity pattern based on $S$.

## ii. Validation 
We validate the pruned model using the ImageNet-1k Validation Set (50k images).Metric: Top-1 Accuracy drop.Success Criteria: $< 1\%$ accuracy drop compared to the dense model.

### iii. The Metric: Visual Wanda
Unlike Magnitude Pruning (which only looks at $|W|$), we calculate importance based on the input feature norms:
$$\text{Score}_{ij} = |W_{ij}| \times ||X_j||_2$$
Where $||X_j||$ is the L2 norm of the input activations for channel $j$, aggregated across a calibration set of images. This ensures we keep weights that handle "loud" visual features.

### iv. The Constraint: 2:4 Structured Sparsity
Instead of pruning the bottom $k\%$ globally, we operate on **local groups**:
1.  Reshape the Weight Matrix $W$ into groups of 4 consecutive elements.
2.  Calculate the Wanda Score for all 4 elements.
3.  **Mask** the 2 elements with the lowest scores in that group.
4.  Result: A mask that is mathematically guaranteed to be 50% sparse and hardware-compliant.

---

## 🛠️ Tech Stack
* **Core:** Python, PyTorch
* **Models:** HuggingFace `transformers` (ViT-Base, ViT-Large)
* **Data:** ImageNet-1k (Calibration samples), COCO
* **Acceleration (Future):** NVIDIA TensorRT, vLLM


## 📂 Project Structure

```bash
VisWanda-Eco-VLM/
├── data/
│   └── calibration_images/    # Small set of 128 images for activation stats
├── src/
│   ├── pruner.py              # Core logic for Wanda & 2:4 Masking
│   ├── model_utils.py         # ViT loading and hook registration
│   └── layer_analysis.py      # Visualizing weight distributions
├── experiments/
│   ├── benchmark_sparsity.py  # Script to measure accuracy drop vs sparsity
│   └── visualize_masks.ipynb  # Notebook to see the 2:4 pattern
├── README.md
└── requirements.txt