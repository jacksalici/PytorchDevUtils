import wandb
import json


class Logger:
    def __init__(self, project_name, avoid_wandb, print_debug = False):
        self.print_debug = print_debug
        self.avoid_wandb = avoid_wandb

        if not self.avoid_wandb:
            wandb.init(project=project_name)
    
    def print_config(self, args):
        if not self.avoid_wandb:
            wandb.config.update(args)
        print(json.dumps(args, indent=4))

    def log(self, info: dict, debug: bool = False):
        if not self.avoid_wandb:
            wandb.log(info)
        
        if not debug or (debug and self.print_debug):
            for key, value in info.items():
                print(f"{key}: {value}", end=" | ")
            print()