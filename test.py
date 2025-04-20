from pytorch_dev_utils.logger import Logger
from pytorch_dev_utils.shapehook import ShapeHook

import torch
from torch import nn


if __name__ == "__main__":
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 16 * 16, 10)
    )
    
    hook_manager = ShapeHook()
    logger = Logger(print_log_level=Logger.LogLevel.INFO)
    # logger.initialize_wandb("test_project")
    logger.log({"message": "Starting model forward pass"}, log_level=Logger.LogLevel.DEBUG)
    
    hook_manager.register_hooks(model, one_time=True)
    
    dummy_input = torch.randn(1, 3, 32, 32)
    
    output = model(dummy_input)
    logger.log({"message": "Model forward pass completed"}, log_level=Logger.LogLevel.INFO)
    
