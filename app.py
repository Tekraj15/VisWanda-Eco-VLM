import gradio as gr
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from peft import PeftModel

# Global state
model = None
processor = None
device = None

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_optimized_model():
    global model, processor, device
    
    # Path where run_pipeline.py saved the model
    # Note: LoRA saves adapters separately, so we load base + adapters
    model_path = "./checkpoints/viswanda_final_model"
    base_model_name = "google/vit-base-patch16-224"
    
    if model is None:
        print(f"⏳ Loading optimized model from {model_path}...")
        device = get_device()
        
        # 1. Load Base Model
        base_model = ViTForImageClassification.from_pretrained(base_model_name)
        
        # 2. Load LoRA Adapters (The Fine-Tuned Part)
        # If you saved the merged model, just load via from_pretrained.
        # If you saved PEFT adapters, load like this:
        try:
            model = PeftModel.from_pretrained(base_model, model_path)
            print("✅ Loaded LoRA adapters.")
        except:
            print("⚠️ Could not load adapters, loading standard model (or full save).")
            model = ViTForImageClassification.from_pretrained(model_path)

        processor = ViTImageProcessor.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
    return "Model Loaded! Ready for Inference."

def predict(image):
    global model, processor, device
    
    if model is None:
        load_optimized_model()
        
    if image is None:
        return None
    
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    top_probs, top_indices = torch.topk(probs, 5)
    
    results = {}
    for score, idx in zip(top_probs[0], top_indices[0]):
        # Handle cases where config might be wrapped by PEFT
        config = model.config if hasattr(model, "config") else model.base_model.config
        label = config.id2label[idx.item()]
        results[label] = float(score)
        
    return results

# Gradio Interface
with gr.Blocks(title="VisWanda Eco-VLM Demo") as demo:
    gr.Markdown("# 👁️🗜️ VisWanda-Eco-VLM: Final Optimized Model")
    gr.Markdown("Running the **Fine-Tuned (LoRA)** 2:4 Pruned ViT.")
    
    status_output = gr.Textbox(label="System Status", value="Waiting...", interactive=False)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            classify_btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            label_output = gr.Label(num_top_classes=5, label="Predictions")
    
    demo.load(load_optimized_model, outputs=status_output)
    classify_btn.click(predict, inputs=input_image, outputs=label_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)