import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from model_utils import get_model
from data_loader import get_calibration_data
from pruner import WandaPruner
from torch.utils.data import DataLoader, TensorDataset

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if isinstance(batch, dict):
                inputs = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
            elif isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
                if len(batch) > 1:
                    labels = batch[1].to(device)
                else:
                    labels = torch.zeros(inputs.shape[0], device=device)
            else:
                # Assuming tensor
                inputs = batch.to(device)
                # Dummy labels if not provided, just for throughput check
                labels = torch.zeros(inputs.shape[0], device=device)

            outputs = model(inputs)
            logits = outputs.logits
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0
    accuracy = 100 * correct / total
    return accuracy

def check_sparsity(model):
    total_params = 0
    zero_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            total_params += weight.numel()
            zero_params += (weight == 0).sum().item()
            
    print(f"Global Sparsity (Linear Layers): {100 * zero_params / total_params:.2f}%")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    # Load Model
    model, processor = get_model()
    model.to(device)

    # Check Baseline Sparsity
    print("Baseline Sparsity:")
    check_sparsity(model)

    # Prepare Calibration Data
    print("Fetching calibration data...")
    # Using imagenet-1k as requested, but small sample for speed in this test run
    calib_images = get_calibration_data(processor, dataset_name="imagenet-1k", n_samples=32) 
    calib_dataset = TensorDataset(calib_images)
    calib_loader = DataLoader(calib_dataset, batch_size=8, shuffle=False)

    # Initialize Pruner
    pruner = WandaPruner(model)
    
    # Prepare Calibration (Forward Pass to collect activations)
    pruner.prepare_calibration(calib_loader, device)
    
    # Prune
    pruner.prune()
    
    # Check Pruned Sparsity
    print("Sparsity after Pruning:")
    check_sparsity(model)
    
    # Verify 2:4 pattern on a random layer
    print("\nVerifying 2:4 pattern on a sample layer...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            # Check first group of 4
            if weight.shape[1] >= 4:
                group = weight[0, :4]
                print(f"Layer: {name}")
                print(f"First group of 4 weights: {group}")
                zeros = (group == 0).sum().item()
                print(f"Zeros in group: {zeros}")
                if zeros == 2:
                    print("2:4 pattern verified for this group.")
                else:
                    print("2:4 pattern FAILED for this group.")
            break

if __name__ == "__main__":
    main()
