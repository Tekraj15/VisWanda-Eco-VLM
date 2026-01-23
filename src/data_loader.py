import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import ViTImageProcessor
import math
import os
import gc
import random
import time

# REMOVED HF_TRANSFER ---
if "HF_HUB_ENABLE_HF_TRANSFER" in os.environ:
    del os.environ["HF_HUB_ENABLE_HF_TRANSFER"]

class SimpleDataset(Dataset):
    def __init__(self, pixel_values, labels):
        self.pixel_values = pixel_values
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "pixel_values": self.pixel_values[idx],
            "labels": self.labels[idx]
        }

def get_train_val_data(processor, n_train=16384, n_val=2048, cache_dir="./data/train_val_dataset"):
    """
    Loads ImageNet-1k using Modulo Striding with Network Robustness.
    """
    cache_file = os.path.join(cache_dir, f"train_{n_train}_val_{n_val}.pt")
    
    # 1. Load Cache if exists
    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}...")
        try:
            data = torch.load(cache_file)
            train_ds = SimpleDataset(data['train_pixels'], data['train_labels'])
            eval_ds = SimpleDataset(data['val_pixels'], data['val_labels'])
            print(f"Loaded {len(train_ds)} train, {len(eval_ds)} val samples from cache.")
            return train_ds, eval_ds
        except Exception as e:
            print(f"Cache corrupted ({e}), regenerating...")
    
    # 2. Setup Collection
    print("No valid cache. Initializing stream...")
    os.makedirs(cache_dir, exist_ok=True)
    
    dataset_name = "imagenet-1k"
    total_images_approx = 1281167
    total_needed = n_train + n_val
    stride = math.floor(total_images_approx / total_needed)
    
    print(f"Pre-allocating tensor storage for {total_needed} images (~11GB RAM)...")
    
    final_pixels = torch.empty((total_needed, 3, 224, 224), dtype=torch.float32)
    final_labels = torch.empty((total_needed,), dtype=torch.long)
    
    # Enable explicit streaming with relaxed verification
    dataset = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True, verification_mode="no_checks")
    
    batch_images = []
    batch_labels = []
    batch_size = 64
    
    current_idx = 0
    stream_idx = 0  # Manual counter since we aren't using enumerate
    
    print(f"Streaming with Stride {stride} (Robust Mode)...")
    
    # --- Fix: ROBUST ITERATOR ---
    # use manual iteration + try/except to survive network hiccups
    iterator = iter(dataset)
    
    while current_idx < total_needed:
        try:
            # Attempt to get next sample
            sample = next(iterator)
            stream_idx += 1
            
            # Check Stride
            if stream_idx % stride == 0:
                batch_images.append(sample['image'].convert("RGB"))
                batch_labels.append(sample['label'])
                
                # Process Batch
                if len(batch_images) >= batch_size:
                    inputs = processor(batch_images, return_tensors="pt")['pixel_values']
                    labels = torch.tensor(batch_labels)
                    
                    batch_len = inputs.shape[0]
                    end_idx = current_idx + batch_len
                    
                    if end_idx > total_needed:
                        trim = total_needed - current_idx
                        inputs = inputs[:trim]
                        labels = labels[:trim]
                        batch_len = trim
                        end_idx = total_needed

                    final_pixels[current_idx:end_idx] = inputs
                    final_labels[current_idx:end_idx] = labels
                    
                    current_idx = end_idx
                    
                    del inputs, labels, batch_images, batch_labels
                    batch_images = []
                    batch_labels = []
                    gc.collect()
                    
                    if current_idx % 1000 == 0:
                        print(f"  Filled {current_idx}/{total_needed} slots...")

        except StopIteration:
            print("End of dataset stream reached early.")
            break
        except Exception as e:
            # Catch network errors (BrokenPipe, ReadTimeout) and continue
            print(f"⚠️ Network Warning at index {stream_idx}: {e}. Skipping sample...")
            # Optional: Sleep briefly to let network recover
            time.sleep(0.5)
            continue
    
    # Handle remaining
    if batch_images and current_idx < total_needed:
        inputs = processor(batch_images, return_tensors="pt")['pixel_values']
        labels = torch.tensor(batch_labels)
        batch_len = min(inputs.shape[0], total_needed - current_idx)
        final_pixels[current_idx : current_idx + batch_len] = inputs[:batch_len]
        final_labels[current_idx : current_idx + batch_len] = labels[:batch_len]
        current_idx += batch_len
        del inputs, labels, batch_images
        gc.collect()

    print(f"Collection complete. Tensor Shape: {final_pixels.shape}")
    
    # --- SHUFFLING & SPLITTING ---
    print("Shuffling indices...")
    indices = torch.randperm(total_needed)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    
    print("Splitting and Saving...")
    train_pixels = final_pixels[train_indices]
    train_labels = final_labels[train_indices]
    val_pixels = final_pixels[val_indices]
    val_labels = final_labels[val_indices]
    
    # Save cache
    torch.save({
        'train_pixels': train_pixels,
        'train_labels': train_labels,
        'val_pixels': val_pixels,
        'val_labels': val_labels
    }, cache_file)
    print("Cache saved.")
    
    return SimpleDataset(train_pixels, train_labels), SimpleDataset(val_pixels, val_labels)

def get_calibration_data(processor, dataset_name="imagenet-1k", n_samples=512, save_dir=None):
    print(f"Fetching {n_samples} calibration samples...")
    dataset = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
    shuffled = dataset.shuffle(seed=123, buffer_size=1000) 
    
    samples = []
    iterator = iter(shuffled)
    for _ in range(n_samples):
        samples.append(next(iterator))
        
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for i, sample in enumerate(samples[:5]): 
            sample['image'].save(os.path.join(save_dir, f"calib_{i}.png"))
        
    inputs = processor([x['image'].convert("RGB") for x in samples], return_tensors="pt")
    return inputs['pixel_values']

def get_validation_data(processor, dataset_name="imagenet-1k", n_samples=5000):
    dataset = load_dataset(dataset_name, split="validation", streaming=True, trust_remote_code=True)
    shuffled = dataset.shuffle(seed=42, buffer_size=5000)
    
    samples = []
    iterator = iter(shuffled)
    
    print(f"Streaming {n_samples} validation samples...")
    for _ in range(n_samples):
        try:
            samples.append(next(iterator))
        except StopIteration:
            break
            
    print(f"Processing {len(samples)} validation images...")
    batch_size = 100
    all_pixels = []
    all_labels = []
    
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        batch_inputs = processor([x['image'].convert("RGB") for x in batch], return_tensors="pt")
        all_pixels.append(batch_inputs['pixel_values'])
        all_labels.extend([x['label'] for x in batch])
    
    return torch.cat(all_pixels, dim=0), torch.tensor(all_labels)

if __name__ == "__main__":
    print("Testing data loader...")
    try:
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        print("--- Testing Train/Val Loader ---")
        train_ds, val_ds = get_train_val_data(processor, n_train=100, n_val=20)
        print(f"Train Size: {len(train_ds)}, Val Size: {len(val_ds)}")
    except Exception as e:
        print(f"Error: {e}")