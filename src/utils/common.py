# Imports a special class that lets you access dictionary keys like object attributes.
# For example, instead of config['key'], you can write config.key.
from box import ConfigBox 
# Imports the Path class, a modern way to handle file system paths.
from pathlib import Path 
# Imports a custom logger object to print informative messages.
from src.utils.logger import logger
# Imports a specific error from the 'box' library to handle it gracefully.
from box.exceptions import BoxValueError 
import os
import yaml

"""" used for reading yaml files and creating directories """

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its content as a ConfigBox."""
    
    # This block tries to run the code inside it. If an error occurs, 
    # it will jump to one of the 'except' blocks.
    try:
        # Opens the file at the given path. 'with' ensures the file is automatically closed.
        with open(path_to_yaml) as yaml_file:
            # Loads the content of the YAML file into a Python dictionary.
            # 'safe_load' is used for security to prevent running malicious code.
            content = yaml.safe_load(yaml_file)
            # Logs a success message showing which file was loaded.
            logger.info(f"YAML file '{path_to_yaml}' loaded successfully.")
            # Converts the dictionary 'content' into a ConfigBox object and returns it.
            return ConfigBox(content)
            
    # If the YAML file is empty, 'ConfigBox(content)' might raise a BoxValueError.
    # This block catches that specific error.
    except BoxValueError:
        # Raises a new, more user-friendly error message.
        raise ValueError("YAML file is empty")
        
    # This catches any other possible error that might occur (e.g., file not found).
    except Exception as e:
        # Re-raises the original exception 'e' to stop the program and show the error.
        raise e


def create_directories(path_to_directories: list, verbose=True):
    """Creates a list of directories."""
    
    # Loops through each directory path provided in the 'path_to_directories' list.
    for path in path_to_directories:
        # Creates the directory. 'makedirs' can create parent folders too (e.g., 'a/b/c').
        # 'exist_ok=True' prevents an error if the directory already exists.
        os.makedirs(path, exist_ok=True)
        # Checks if the 'verbose' flag is True (it is by default).
        if verbose:
            # If verbose is True, it logs a message confirming the directory was created.
            logger.info(f"Created directory at: {path}")