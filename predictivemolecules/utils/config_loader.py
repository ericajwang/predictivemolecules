import yaml
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class Config:
    model: Dict[str, Any]
    data: Dict[str, Any]
    training: Dict[str, Any]
    optimizer: Dict[str, Any]
    scheduler: Dict[str, Any]
    loss: Dict[str, Any]
    evaluation: Dict[str, Any]


def load_config(config_path: str) -> Config:
    # loads configuration from YAML file
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(**config_dict)


def save_config(config: Config, save_path: str):
    config_dict = {
        'model': config.model,
        'data': config.data,
        'training': config.training,
        'optimizer': config.optimizer,
        'scheduler': config.scheduler,
        'loss': config.loss,
        'evaluation': config.evaluation,
    }
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

