#Full Pipeline for Pruning and LoRA Fine-Tuning Recovery
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import ViTForImageClassification, ViTImageProcessor
from pruner import WandaPruner
from finetune import apply_lora_and_finetune
from data_loader import get_train_val_data, get_calibration_data  

def main():
    print("=== Phase 1: Pruning & Recovery Pipeline ===")
    
    # Device detection with MPS support
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    model_name = "google/vit-base-patch16-224"
    save_path = "./checkpoints/viswanda_final_model"

    # 1. Load Original Model
    print(f"Loading {model_name}...")
    model = ViTForImageClassification.from_pretrained(model_name).to(device)
    processor = ViTImageProcessor.from_pretrained(model_name)
    
    # 2. Load Data (for calibration AND fine-tuning)
    print("Loading datasets...")
    train_ds, eval_ds = get_train_val_data(processor, n_train=4096, n_val=128)

    # 3. Pruning (Wanda 2:4)
    print("Step 3: Pruning...")
    pruner = WandaPruner(model)
    
    # Use get_calibration_data for proper calibration (returns a tensor)
    print("Fetching calibration data...")
    calib_images = get_calibration_data(processor, dataset_name="imagenet-1k", n_samples=128)
    calib_dataset = torch.utils.data.TensorDataset(calib_images)
    calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=8, shuffle=False)
    
    pruner.prepare_calibration(calib_loader, device)
    pruner.prune()

    # 4.1 SAVE PRUNED BASE MODEL and SPARSITY MASKS (before LoRA)
    # This is critical coz PEFT doesn't save base weights!
    pruned_base_path = "./checkpoints/viswanda_pruned_base"
    print(f"Saving pruned base model to {pruned_base_path}...")
    model.save_pretrained(pruned_base_path)
    processor.save_pretrained(pruned_base_path)
    
    # Save sparsity masks (which weights are zero)
    print("Saving sparsity masks...")
    sparsity_masks = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Store boolean mask: True where weight is non-zero
            sparsity_masks[name] = (module.weight.data != 0).cpu()
    
    # 4.2 Recovery (LoRA Fine-Tuning)
    print("Step 4: Recovery Fine-Tuning...")
    model = apply_lora_and_finetune(
        model, 
        train_dataset=train_ds, 
        eval_dataset=eval_ds,
        output_dir="./viswanda_recovery_checkpoints"
    )

    # 5. Merge LoRA into base(pruned weights) and save the FULL model
    print(f"Step 5: Merging LoRA into base and saving to {save_path}...")
    
    # Merge LoRA weights into base model
    merged_model = model.merge_and_unload()
    
    # 5.1 RE-APPLY SPARSITY MASKS (critical fix!)
    # LoRA merge fills in zeros, so we need to re-zero them
    print("Re-applying sparsity masks to preserve pruning...")
    for name, module in merged_model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in sparsity_masks:
            mask = sparsity_masks[name].to(module.weight.device)
            module.weight.data *= mask.float()
    
    # Verify sparsity after re-masking
    total = 0
    zeros = 0
    for name, module in merged_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            total += module.weight.numel()
            zeros += (module.weight.data == 0).sum().item()
    print(f"Final sparsity after re-masking: {100 * zeros / total:.2f}%")
    
    # Save the merged model (includes pruned + LoRA weights)
    merged_model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    
    print("Pipeline Complete. Ready for app.py!")

if __name__ == "__main__":
    main()