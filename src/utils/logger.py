import os
import sys
import logging
import yaml

# Define the format for the log messages
# logging_str = "[%(asctime)s] %(levelname)s - %(module)s - %(message)s"

# 1. YAML file ko padho
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

logging_str = config['logging']['format']
# level = config['logging']['level']
log_dir = config['logging']['log_dir']
log_filename = config['logging']['log_filename']

# Directory aur Filename ko
full_log_path = os.path.join(log_dir, log_filename)
os.makedirs(log_dir, exist_ok=True)

# Create the logger object
logger = logging.getLogger("MLP1Logger")
logger.setLevel(logging.INFO) # Set the level on the logger itself

# --- THIS IS THE IMPORTANT CHECK ---
# Check if the logger already has handlers to avoid duplication
if not logger.handlers:
    # Define where the logs should go
    file_handler = logging.FileHandler(full_log_path)
    console_handler = logging.StreamHandler(sys.stdout)

    # Set the format for the handlers
    formatter = logging.Formatter(logging_str)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)