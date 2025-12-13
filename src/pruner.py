# WandaPruner class and hooks
import torch
import torch.nn as nn

class WandaPruner:
    def __init__(self, model):
        self.model = model
        self.activations = {}

    def add_hook(self, name, module):
        """
        Adds a forward hook to a layer to capture input activations.
        We need the L2 norm of the input X for each column j.
        """
        def hook(module, input, output):
            # input is a tuple, we want the first element
            # Shape: (Batch, Tokens, Features)
            inp = input[0]
            
            # Reshape to (Batch * Tokens, Features) to treat all tokens as samples
            inp = inp.view(-1, inp.shape[-1])
            
            # Calculate L2 norm for each feature column j and accumulate(sum(x_ij^2)) over the calibration set.
            
            if name not in self.activations:
                self.activations[name] = torch.zeros(inp.shape[1], device=inp.device)
                
            # Add sum of squares for this batch
            self.activations[name] += torch.sum(inp ** 2, dim=0)

        return module.register_forward_hook(hook)

    def prepare_calibration(self, dataloader, device):
        """
        Runs the model on calibration data to populate self.activations.
        """
        print("Preparing calibration...")
        hooks = []
        
        # Register hooks for all Linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(self.add_hook(name, module))
                
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                elif isinstance(batch, dict):
                    inputs = batch['pixel_values'].to(device)
                else:
                    inputs = batch.to(device)
                    
                self.model(inputs)
                
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        # Finalize activations: take sqrt to get L2 norm
        for name in self.activations:
            self.activations[name] = torch.sqrt(self.activations[name])
            
        print("Calibration complete.")

    def prune(self):
        """
        Applies Wanda pruning with 2:4 structured sparsity constraint.
        """
        print("Pruning model with Wanda metric and 2:4 sparsity...")
        
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    if name not in self.activations:
                        print(f"Skipping {name}, no activation data found.")
                        continue
                        
                    # 1. Get Weights and Activation Norms
                    W = module.weight.data # Shape: (Out, In)
                    X_norm = self.activations[name] # Shape: (In,)
                    
                    # 2. Compute Wanda Score
                    # Score_ij = |W_ij| * ||X_j||
                    # broadcast X_norm to match W
                    # W: (Out, In), X_norm: (In,) -> Broadcast works automatically
                    
                    # abs() to handle zero issues if needed.
                    W_abs = torch.abs(W)
                    score = W_abs * X_norm
                    
                    # 3. Enforce 2:4 Sparsity
                    # We need to reshape to groups of 4; prune along the 'in' dimension (columns) for 2:4.
                    # So reshape (Out, In) -> (Out, In/4, 4)
                    
                    if W.shape[1] % 4 != 0:
                        print(f"Skipping {name}, input dimension {W.shape[1]} not divisible by 4.")
                        continue
                        
                    out_features, in_features = W.shape
                    
                    # Reshape score to identify bottom 2
                    score_reshaped = score.view(out_features, in_features // 4, 4)
                    
                    # Find indices of the 2 smallest scores in the last dimension
                    _, indices = torch.topk(score_reshaped, k=2, dim=-1, largest=False)
                
                    mask = torch.ones_like(score_reshaped, dtype=torch.bool)
                    mask.scatter_(dim=-1, index=indices, src=torch.zeros_like(mask))
                    mask = mask.view(out_features, in_features)
                    
                    # 4. Apply Mask
                    module.weight.data *= mask.float()
                    
        print("Pruning complete.")
