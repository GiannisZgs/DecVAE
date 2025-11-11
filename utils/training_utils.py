import torch
from safetensors.torch import save_file
import os
from sklearn.metrics import accuracy_score, f1_score
from latent_analysis_utils import calculate_unweighted_accuracy

def multiply_grads(params, c):
    """Multiplies grads by a constant *c*."""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(c):
                c = c.to(p.grad.device)
            p.grad.data.mul_(c)

def get_grad_norm(params, scale=1):
    """Compute grad norm given a gradient scale."""
    total_norm = 0.0
    for name,p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            if (param_norm!=param_norm).any() or torch.isinf(param_norm).any():
                print(name,param_norm)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model_excluding_params(model, save_path, exclude_keys=None):
    """
    Save a model in .safetensors format while excluding specific parameters by name.
    
    Args:
        model: The PyTorch model to save
        save_path: Directory path where to save the model
        exclude_keys: List of parameter names (or partial names) to exclude from saving
    """

    
    
    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Get the state dict
    state_dict = model.state_dict()
    
    # Create a filtered state dict excluding specified parameters
    if exclude_keys:
        filtered_state_dict = {}
        for key, value in state_dict.items():
            # Check if the current key should be excluded
            if not any(exclude_key in key for exclude_key in exclude_keys):
                filtered_state_dict[key] = value
        
        excluded_keys = set(state_dict.keys()) - set(filtered_state_dict.keys())
        print(f"Excluded {len(excluded_keys)} parameters from saving:")
        for key in sorted(excluded_keys):
            print(f"  - {key}")
    else:
        filtered_state_dict = state_dict
    
    # Save the filtered state dict using safetensors
    save_file(filtered_state_dict, os.path.join(save_path, "model.safetensors"))
        
    print(f"Model saved to {save_path}")

def calculate_metrics(logits, labels):
    """
    Calculate accuracy and F1-score from logits and labels.
    
    Args:
        logits: Model predictions (B, S, N) or (B, N)
        labels: Ground truth labels (B, S) or (B)            
    Returns:
        dict: Dictionary with accuracy and F1-score metrics
    """
    
    # Move tensors to CPU and convert to numpy
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()
    
    # Handle different shapes
    if len(logits.shape) == 3:  # [B, S, N]
        # Reshape to [B*S, N]
        logits_flat = logits.reshape(-1, logits.shape[-1])
        # Handle different label shapes
        if len(labels.shape) == 2:  # [B, S]
            labels_flat = labels.reshape(-1)
        else:
            raise ValueError(f"Unsupported label shape: {labels.shape} for logits shape: {logits.shape}")
    else:  # [B, N]
        logits_flat = logits
        labels_flat = labels
    
    # Get predictions
    preds = torch.argmax(logits_flat, dim=-1).numpy()
    labels_np = labels_flat.numpy()
    
    # Filter out padding labels (-100)
    valid_indices = labels_np != -100
    if valid_indices.sum() == 0:
        return {"accuracy": 0.0, "unweighted_accuracy": 0.0, "f1_score": 0.0}
    
    preds = preds[valid_indices]
    labels_np = labels_np[valid_indices]
    
    # Calculate metrics
    accuracy = accuracy_score(labels_np, preds)
    unweighted_accuracy = calculate_unweighted_accuracy(labels_np, preds)
    weighted_f1 = f1_score(labels_np, preds, average='weighted')
    
    return {
        "accuracy": accuracy,
        "unweighted_accuracy": unweighted_accuracy,
        "f1_score": weighted_f1
    }

class EarlyStopping:
    def __init__(self, patience=5, min_delta_percent=0.2, min_steps = 100000):
        self.patience = patience
        self.min_delta_percent = min_delta_percent 
        self.min_steps = min_steps
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss, steps):
        
        if val_loss < self.best_loss:
            improvement = (self.best_loss - val_loss) / self.best_loss
        else:
            improvement = 0

        if steps < self.min_steps:
            return False
        else:
            if improvement >= self.min_delta_percent:
                self.best_loss = val_loss
                self.counter = 0  # Reset counter if improvement
            else:
                self.counter += 1  # Increment counter if no improvement
                if self.counter >= self.patience:
                    self.early_stop = True

