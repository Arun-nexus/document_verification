import yaml
import os
from pathlib import Path
from logger.log_config import setup_logger

logger = setup_logger

def load_params(params_path = "params.yaml" ):
    if params_path is None:
        logger.error(f"{params_path} is not created or may created by different name")
        return
    else:
        with open(params_path,"r") as f:
            params = yaml.safe_load(f)
        return params
         
