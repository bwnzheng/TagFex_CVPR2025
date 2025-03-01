import yaml
from pathlib import Path

def load_yaml(p):
    with Path(p).open('r') as f:
        content = yaml.safe_load(f)
    return content

def load_configs(config_list):
    configs = dict()
    for config in config_list:
        configs.update(load_yaml(config))
    return configs