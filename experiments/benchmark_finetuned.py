# Script to evaluate the fine-tuned pruned model
# This loads the saved LoRA-finetuned model instead of pruning from scratch
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from transformers import ViTForImageClassification, ViTImageProcessor
from peft import PeftModel
from data_loader import get_validation_data
from torch.utils.data import DataLoader, TensorDataset

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
                labels = batch[1].to(device) if len(batch) > 1 else torch.zeros(inputs.shape[0], device=device)
            else:
                inputs = batch.to(device)
                labels = torch.zeros(inputs.shape[0], device=device)

            outputs = model(inputs)
            logits = outputs.logits
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0
    return 100 * correct / total

def check_sparsity(model):
    """Check sparsity in the base model (underneath LoRA adapters if present)."""
    total_params = 0
    zero_params = 0
    
    # Handle PEFT wrapped models vs regular models
    # PeftModel has .base_model.model structure and Regular ViTForImageClassification should be used directly
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        # PEFT model
        target_model = model.base_model.model
    else:
        # Regular model - use it directly
        target_model = model
    
    for name, module in target_model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            total_params += weight.numel()
            zero_params += (weight == 0).sum().item()
    
    sparsity = 100 * zero_params / total_params if total_params > 0 else 0
    return sparsity

def main():
    device = get_device()
    print(f"Using device: {device}")
    
    model_path = "./checkpoints/viswanda_final_model"
    base_model_name = "google/vit-base-patch16-224"
    
    # 1. BASELINE EVALUATION
    print("\n" + "="*50)
    print("BASELINE (Original ViT-Base)")
    print("="*50)
    
    base_model = ViTForImageClassification.from_pretrained(base_model_name).to(device)
    processor = ViTImageProcessor.from_pretrained(base_model_name)
    
    # INCREASE validation sample size to 1024 for statistical significance
    print("Loading validation data (n=1024)...")
    val_images, val_labels = get_validation_data(processor, dataset_name="imagenet-1k", n_samples=2048)
    val_dataset = TensorDataset(val_images, val_labels)
    # Batch size 32 is fine for M1/M2 inference
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Evaluating baseline on {len(val_labels)} samples...")
    baseline_accuracy = evaluate(base_model, val_loader, device)
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")
    print(f"Baseline Sparsity: {check_sparsity(base_model):.2f}%")
    
    # Free memory
    del base_model
    torch.mps.empty_cache() if device.type == "mps" else None
    
    # 2. FINETUNED MODEL EVALUATION
    print("\n" + "="*50)
    print("FINETUNED (Pruned + LoRA Merged)")
    print("="*50)
    
    # Load the fine-tuned model (now saved as merged full model)
    print(f"Loading fine-tuned model from {model_path}...")
    
    # Load as full model (LoRA merged into pruned base)
    finetuned_model = ViTForImageClassification.from_pretrained(model_path)
    print("✅ Loaded merged model (pruned + LoRA)")
    
    finetuned_model.to(device)
    finetuned_model.eval()
    
    # Check sparsity
    sparsity = check_sparsity(finetuned_model)
    print(f"Model Sparsity: {sparsity:.2f}%")
    
    # Evaluate
    print(f"Evaluating fine-tuned model on {len(val_labels)} samples...")
    finetuned_accuracy = evaluate(finetuned_model, val_loader, device)
    print(f"Fine-tuned Accuracy: {finetuned_accuracy:.2f}%")
    
    # 3. COMPARISON
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    drop = baseline_accuracy - finetuned_accuracy
    
    print(f"┌────────────────────────────────────────────┐")
    print(f"│ Foundation Model Accuracy:   {baseline_accuracy:>6.2f}% │")
    print(f"│ Pruned + LoRA Fine-tuned Accuracy: {finetuned_accuracy:>6.2f}% │")
    print(f"│ Accuracy Drop:       {drop:>6.2f}%                │")
    print(f"│ Global Sparsity:     {sparsity:>6.2f}%             │")
    print(f"└────────────────────────────────────────────┘")
    
    if drop < 1.0:
        print("🎉 SUCCESS: Accuracy drop is within 1% tolerance!")
    elif drop < 2.0:
        print("✓ GOOD: Accuracy drop is within 2%.")
    elif drop < 5.0:
        print("⚠️ WARNING: Accuracy drop is between 2-5%.")
    else:
        print("❌ FAIL: Accuracy drop exceeds 5%.")
    
    # Confidence check
    print("\nConfidence check on first 5 samples...")
    with torch.no_grad():
        inputs = val_images[:5].to(device)
        outputs = finetuned_model(inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top_probs, _ = torch.max(probs, dim=-1)
        print(f"Top Confidence Scores: {top_probs.cpu().numpy()}")
        
        if (top_probs > 0.5).all():
            print("Model is confident in its predictions.")
        else:
            print("⚠️ Low confidence detected.")

if __name__ == "__main__":
    main()
