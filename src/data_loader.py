import torch
from datasets import load_dataset
from transformers import ViTImageProcessor

import os
from PIL import Image

def get_calibration_data(processor, dataset_name="imagenet-1k", n_samples=128, save_dir=None):
    """
    Streams only the first n_samples from the dataset. No massive download required.
    Saves the raw images in the save dir.
    """
    print(f"Streaming {n_samples} images from {dataset_name}...")
    
    # "streaming=True" is the magic keyword
    if dataset_name == "imagenet-1k":
        # Requires 'huggingface-cli login' in terminal first
        dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
    elif dataset_name == "coco":
        dataset = load_dataset("detection-datasets/coco", split="train", streaming=True)
    else:
        # Fallback for testing without login
        print(f"Dataset {dataset_name} not explicitly handled, trying generic load...")
        dataset = load_dataset(dataset_name, split="train", streaming=True)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving images to {save_dir}...")

    calibration_batch = []
    
    # Iterating through the stream
    for i, sample in enumerate(dataset):
        if i >= n_samples:
            break
            
        # Get image (PIL format)
        image = sample['image']
        
        # Save raw image if requested
        if save_dir:
            image_path = os.path.join(save_dir, f"calib_{i:04d}.jpg")
            image.save(image_path)
        
        # Convert grayscale to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Preprocess for the model (returns pytorch tensor)
        # return_tensors="pt" gives [1, 3, 224, 224]
        inputs = processor(images=image, return_tensors="pt")
        calibration_batch.append(inputs['pixel_values'])
    
    # Concatenate into one massive tensor [128, 3, 224, 224]
    if not calibration_batch:
        raise ValueError("No images collected! Check dataset connection or name.")
        
    return torch.cat(calibration_batch, dim=0)

# For Brain Function" test: To calculate the accuracy of the model on the validation and perform the qualitative confidence check set
def get_validation_data(processor, dataset_name="imagenet-1k", n_samples=128):
    """
    Streams n_samples from the validation split and returns (images, labels).
    """
    print(f"Streaming {n_samples} validation images from {dataset_name}...")
    
    split = "validation"
    if dataset_name == "imagenet-1k":
        dataset = load_dataset("ILSVRC/imagenet-1k", split=split, streaming=True)
    elif dataset_name == "coco":
        # Fallback to train if validation not found or handle differently.
        dataset = load_dataset("detection-datasets/coco", split=split, streaming=True)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    images_batch = []
    labels_batch = []
    
    for i, sample in enumerate(dataset):
        if i >= n_samples:
            break
            
        image = sample['image']
        label = sample['label']
        
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        inputs = processor(images=image, return_tensors="pt")
        images_batch.append(inputs['pixel_values'])
        labels_batch.append(label)
    
    if not images_batch:
        raise ValueError("No validation images collected!")
        
    images = torch.cat(images_batch, dim=0)
    labels = torch.tensor(labels_batch)
    
    return images, labels


def get_train_val_data(processor, dataset_name="imagenet-1k", n_train=1000, n_val=128):
    """
    Creates Hugging Face datasets for training and validation.
    Returns datasets with proper format for HF Trainer.
    """
    from datasets import Dataset
    
    print(f"Preparing train ({n_train}) and val ({n_val}) datasets from {dataset_name}...")
    
    def load_split_as_hf_dataset(split, n_samples):
        """Helper to load a split as HF Dataset with proper format."""
        if dataset_name == "imagenet-1k":
            stream = load_dataset("ILSVRC/imagenet-1k", split=split, streaming=True)
        else:
            stream = load_dataset(dataset_name, split=split, streaming=True)
        
        pixel_values_list = []
        labels_list = []
        
        for i, sample in enumerate(stream):
            if i >= n_samples:
                break
                
            image = sample['image']
            label = sample['label']
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Process image
            inputs = processor(images=image, return_tensors="pt")
            pixel_values_list.append(inputs['pixel_values'].squeeze(0))
            labels_list.append(label)
        
        # Stack into tensors
        pixel_values = torch.stack(pixel_values_list)
        labels = torch.tensor(labels_list)
        
        # Create HF Dataset
        return Dataset.from_dict({
            "pixel_values": pixel_values,
            "labels": labels
        })
    
    train_ds = load_split_as_hf_dataset("train", n_train)
    val_ds = load_split_as_hf_dataset("validation", n_val)
    
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    return train_ds, val_ds

if __name__ == "__main__":
    import sys
    # Test the loader
    print("Testing data loader...")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    # Use a public dataset for the test to avoid login issues, or use imagenet-1k if logged in. 
    try:
        # Save to data/calibration_images as requested
        save_path = os.path.join(os.path.dirname(__file__), "../data/calibration_images")
        calib_data = get_calibration_data(processor, dataset_name="imagenet-1k", n_samples=128, save_dir=save_path)
        print(f"Calibration Tensor Shape: {calib_data.shape}")
        print(f"Images saved to {save_path}")
        
    except Exception as e:
        print(f"Error testing with imagenet-1k: {e}")
        print("Please ensure you are logged in with `huggingface-cli login`.")
    
    print("Done.")
    sys.exit(0)