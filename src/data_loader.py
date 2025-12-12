import torch
from datasets import load_dataset
from transformers import ViTImageProcessor

def get_calibration_data(processor, dataset_name="imagenet-1k", n_samples=128):
    """
    Streams only the first n_samples from the dataset. No massive download required.
    """
    print(f"Streaming {n_samples} images from {dataset_name}...")
    
    # "streaming=True" is the magic keyword
    if dataset_name == "imagenet-1k":
        dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, trust_remote_code=True)
    elif dataset_name == "coco":
        dataset = load_dataset("detection-datasets/coco", split="train", streaming=True, trust_remote_code=True)
    else:
        # Fallback for testing without login
        print(f"Dataset {dataset_name} not explicitly handled, trying generic load...")
        dataset = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
    
    calibration_batch = []
    
    # Iterating through the stream
    for i, sample in enumerate(dataset):
        if i >= n_samples:
            break
            
        # Get image (PIL format)
        image = sample['image']
        
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

if __name__ == "__main__":
    # Test the loader
    print("Testing data loader...")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    # Use a public dataset for the test to avoid login issues during quick check
    # Or use imagenet-1k if logged in. 
    try:
        calib_data = get_calibration_data(processor, dataset_name="imagenet-1k", n_samples=8)
        print(f"Calibration Tensor Shape: {calib_data.shape}")
        

    except Exception as e:
        print(f"Error testing with imagenet-1k: {e}")
        print("Please ensure you are logged in with `huggingface-cli login`.")