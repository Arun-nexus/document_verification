import logging
import os
from datetime import datetime

def setup_logger(log_name="document verification"):
    os.makedirs("logs",exist_ok=True)
    log_file = f"logs/{log_name}_{datetime.now().strftime('%y%m%d')}.log"

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file) 
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger
