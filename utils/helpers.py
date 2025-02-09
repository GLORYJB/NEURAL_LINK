# utils/helpers.py

import os

def validate_file_path(file_path):
    """
    Validate that a file exists at the given path.

    Parameters:
        file_path (str): Path to the file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.exists(file_path)