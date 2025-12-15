import gradio as gr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_utils import get_model
from data_loader import get_calibration_data
from pruner import WandaPruner

# Global state
model = None
processor = None
device = None
is_pruned = False

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def check_sparsity(model):
    total_params = 0
    zero_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            total_params += weight.numel()
            zero_params += (weight == 0).sum().item()
    return 100 * zero_params / total_params

def initialize_and_prune():
    global model, processor, device, is_pruned
    
    if model is None:
        print("⏳ Initializing model...")
        device = get_device()
        model, processor = get_model()
        model.to(device)
        
        # Fix 1: Set model to EVAL mode immediately to disables Dropout and Batch Norm updates.
        model.eval() 
    
    if not is_pruned:
        print("Starting Pruning Process...")
        try:
            print("Fetching calibration data...")
            # Note: Ensure data_loader.py accepts 'processor' if you modified it, 
            # otherwise remove 'processor' arg to match original definition.
            calib_images = get_calibration_data(processor, dataset_name="imagenet-1k", n_samples=128)
            calib_dataset = TensorDataset(calib_images)
            calib_loader = DataLoader(calib_dataset, batch_size=8, shuffle=False)
            
            # Prune
            pruner = WandaPruner(model)
            pruner.prepare_calibration(calib_loader, device)
            pruner.prune()
            
            is_pruned = True
            print("Pruning Complete!")
            
        except Exception as e:
            print(f"Pruning failed: {e}")
            return f"Error during pruning: {str(e)}"

    sparsity = check_sparsity(model)
    return f"Model Ready! Global Sparsity: {sparsity:.2f}% (Target: ~33% for Hybrid 2:4)"

def predict(image):
    global model, processor, device
    
    if model is None:
        initialize_and_prune()
        
    if image is None:
        return None
    
    # Fix 2: Handle PNG/RGBA inputs
    # ViT expects 3 channels (RGB). Screenshots often have 4 (RGBA).
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Ensure Eval mode is active during inference
    model.eval()
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    top_probs, top_indices = torch.topk(probs, 5)
    
    results = {}
    for score, idx in zip(top_probs[0], top_indices[0]):
        label = model.config.id2label[idx.item()]
        results[label] = float(score)
        
    return results

# Define Gradio Interface
with gr.Blocks(title="VisWanda Eco-VLM Demo") as demo:
    gr.Markdown("# 👁️🗜️🧠 VisWanda-Eco-VLM: Pruned Vision Transformer Demo")
    gr.Markdown("""
    This demo showcases a **Vision Transformer (ViT-Base)** compressed using **Wanda Pruning (Hybrid 2:4)**.
    
    **Why is it confident?**
    We use **Safe Mode pruning** (skipping Attention layers), retaining ~96% accuracy of ViT-Base on ImageNet-1k.
    """)
    
    status_output = gr.Textbox(label="System Status", value="Waiting to initialize...", interactive=False)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            classify_btn = gr.Button("Classify", variant="primary")
        
        with gr.Column():
            label_output = gr.Label(num_top_classes=5, label="Predictions")
    
    # Event handlers
    demo.load(initialize_and_prune, outputs=status_output)
    classify_btn.click(predict, inputs=input_image, outputs=label_output)

if __name__ == "__main__":
    demo.launch(share=False)