# ViT-Base loading and hook registration
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader

def get_model(model_name="google/vit-base-patch16-224"):
    """
    Loads a Vision Transformer model and its processor.
    """
    print(f"Loading model: {model_name}")
    model = ViTForImageClassification.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    return model, processor

def get_calibration_data(processor, n_samples=128, dataset_name="imagenet-1k", split="train"):
    """
    Loads a calibration dataset. 
    
    For this implementation, we'll use 'imagenette' (a smaller subset of imagenet) as a default proxy 
    to ensure it runs out of the box, or 'cifar100' resized.
    Let's use 'frgfm/imagenette' which is standard for these tests.
    """
    print(f"Loading calibration data from {dataset_name}...")
    
    try:
        # Try loading a small subset of ImageNet-1k if available, or fall back to Imagenette
        dataset = load_dataset("frgfm/imagenette", "160px", split=split, streaming=True)
    except Exception as e:
        print(f"Could not load imagenette: {e}. Falling back to CIFAR-100.")
        dataset = load_dataset("cifar100", split=split, streaming=True)

    # Prepare data
    data = []
    count = 0
    for item in dataset:
        if count >= n_samples:
            break
        
        image = item['image']
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        inputs = processor(images=image, return_tensors="pt")
        data.append(inputs['pixel_values'])
        count += 1
        
    print(f"Collected {len(data)} calibration samples.")
    return torch.cat(data, dim=0)
