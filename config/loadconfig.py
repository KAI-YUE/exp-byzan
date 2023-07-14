import yaml
import os

def load_config(config_path=""):
    """Load configurations of yaml file"""
    current_path = os.path.dirname(__file__)

    if os.path.exists(config_path):
        with open(config_path, "r") as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)
    else:
        with open(os.path.join(current_path, "config.yaml"), "r") as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)

    # Empty class for yaml loading
    class cfg: pass
    
    for key in config:
        setattr(cfg, key, config[key])

    return cfg
