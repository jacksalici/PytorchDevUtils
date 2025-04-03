import torch
import torch.nn as nn

def register_shape_hooks(model):
    """
    Register forward hooks on all modules of a model to print tensor shapes.
    
    Args:
        model (nn.Module): PyTorch model to register hooks on
    
    Returns:
        list: List of hook handles that can be removed with handle.remove()
    """
    hooks = []
    
    def hook_fn(module, input, output):
        module_name = module.__class__.__name__
        
        if isinstance(output, torch.Tensor):
            print(f"{module_name} output shape: {output.shape}")
        elif isinstance(output, tuple) and all(isinstance(o, torch.Tensor) for o in output):
            shapes = [o.shape for o in output]
            print(f"{module_name} output shapes: {shapes}")
        else:
            print(f"{module_name} output type: {type(output)}")
    
    for name, module in model.named_modules():
        if name == '':
            continue
        
        handle = module.register_forward_hook(hook_fn)
        hooks.append(handle)
    
    return hooks

if __name__ == "__main__":
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 16 * 16, 10)
    )
    
    handles = register_shape_hooks(model)
    
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    
    for handle in handles:
        handle.remove()