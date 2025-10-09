"""
Load environment variables from .env or .env.example file.
"""

import os

from dotenv import find_dotenv, load_dotenv


def load_env():
    """
    Load environment variables from .env or .env.example file.
    """
    if os.path.exists(".env"):
        load_dotenv(find_dotenv(".env"))
    elif os.path.exists(".env.example"):
        load_dotenv(find_dotenv(".env.example"))
