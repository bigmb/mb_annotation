## extra functions

import os
from dotenv import load_dotenv

__all__ = ["load_env_file"]

def load_env_file(file_path='.env'):
    """
    Load environment variables from a .env file.

    Args:
        file_path (str): Path to the .env file. Defaults to '.env'.

    Returns:
        None
    """
    load_dotenv(file_path)

    # Get the loaded environment variables
    env_vars = os.environ

    return env_vars