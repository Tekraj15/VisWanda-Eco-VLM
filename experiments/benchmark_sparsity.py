# Script to measure accuracy drop vs sparsity
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from model_utils import get_model
from data_loader import get_calibration_data, get_validation_data
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

    # --- NEW: Baseline Accuracy Check ---
    print("\n------------------------------------------------")
    print("📊 BASELINE CHECK")
    print("------------------------------------------------")
    try:
        val_images, val_labels = get_validation_data(processor, dataset_name="imagenet-1k", n_samples=128)
        val_dataset = TensorDataset(val_images, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        print(f"Evaluating baseline accuracy on {len(val_labels)} validation samples...")
        baseline_accuracy = evaluate(model, val_loader, device)
        print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")
    except Exception as e:
        print(f"Skipping baseline check due to error: {e}")
        baseline_accuracy = None

    # Check Baseline Sparsity
    print("\nBaseline Sparsity:")
    check_sparsity(model)

    # Prepare Calibration Data
    print("\nFetching calibration data...")
    # Increased to 128 as per Wanda recommendation
    calib_images = get_calibration_data(processor, dataset_name="imagenet-1k", n_samples=128) 
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

    # --- NEW: Accuracy Verification ---
    print("\n------------------------------------------------")
    print("🏥 SANITY CHECK: Is the model still smart?")
    print("------------------------------------------------")

    try:
        # 1. Load a small validation set
        val_images, val_labels = get_validation_data(processor, dataset_name="imagenet-1k", n_samples=128)
        
        # Create valid dataloader
        val_dataset = TensorDataset(val_images, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 2. Evaluate Accuracy
        print(f"Evaluating post-pruning accuracy on {len(val_labels)} validation samples...")
        accuracy = evaluate(model, val_loader, device)
        print(f"Post-Pruning Accuracy: {accuracy:.2f}%")
        
        if baseline_accuracy is not None:
            drop = baseline_accuracy - accuracy
            print(f"Accuracy Drop: {drop:.2f}%")
            if drop < 1.0:
                print("✅ SUCCESS: Accuracy drop is within 1% tolerance.")
            else:
                print("⚠️ WARNING: Accuracy drop is larger than 1%.")
        
        # 3. Confidence Check (Qualitative)
        print("\nChecking confidence on first 5 samples...")
        model.eval()
        with torch.no_grad():
            inputs = val_images[:5].to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            top_probs, top_indices = torch.max(probs, dim=-1)
            
            print(f"Top Confidence Scores: {top_probs.cpu().numpy()}")
            
            if (top_probs > 0.5).all():
                 print("✅ PASS: Model is confident in its predictions.")
            else:
                 print("⚠️ WARNING: Model confidence is low. Pruning might have been too aggressive.")
                 
    except Exception as e:
        print(f"Skipping accuracy check due to error: {e}")
        print("Ensure you have access to the validation split of the dataset.")

if __name__ == "__main__":
    main()
