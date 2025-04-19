import json
from enum import Enum


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

    def __str__(self):
        return self.name


class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, project_name=None, avoid_wandb=False, print_log_level=LogLevel.INFO):
        if not hasattr(self, "_initialized"):
            self.print_log_level = print_log_level
            self.avoid_wandb = avoid_wandb

            if not self.avoid_wandb and project_name:
                self._initialize_wandb(project_name)
            
            self._initialized = True

    def _initialize_wandb(self, project_name):
        import wandb
        wandb.init(project=project_name)
        self._wandb_available = True    
    
    def print_config(self, args):
        if 'wandb' in globals() and wandb.run is not None:
            wandb.config.update(args)

    def log(self, log_data: dict, log_level: LogLevel = LogLevel.INFO):
        """
        Logs the provided information.

        Parameters:
        - log_data (dict): A dictionary containing the information to log.
        - log_level (LogLevel): The log level for the message. Default is LogLevel.INFO.

        Behavior:
        - Logs the information to the console and, if enabled, to the wandb service.
        - The log level determines if the log is printed in the logs.
        """            
                    
        if not self.avoid_wandb and getattr(self, "_wandb_available", False):
            wandb.log(log_data)
        
        if log_level.value >= self.print_log_level.value:
            print(json.dumps(log_data, indent=4))
        
