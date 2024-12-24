"""
Logger functions.
"""

import logging
from colorlog import ColoredFormatter

def init_logger(filename='app.log'):
    """
    Method for initializing logs.

    Creates and initializes a logger with specified format, logging messages into
    a specified file and console. If the root logger already has handlers, it removes them to avoid duplicate logs.
    
    Parameters
    ----------
    filename : str, optional
        The name of the log file to which the logger will write the logs, by default 'app.log'

    Notes
    -----
    The format of each logging message: {timestamp} : {level name} : {filename} - %(message)s

    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Create a file handler
    file_handler = logging.FileHandler(filename)
    format = logging.Formatter('%(asctime)s : %(levelname)s : %(filename)s - %(message)s')
    file_handler.setFormatter(format)

    # Make sure root_logger has not been set up before
    if root_logger.hasHandlers():
           root_logger.handlers.clear()

    root_logger.addHandler(file_handler)

    # Create a stream handler with custom color formatting
    stream_handler = logging.StreamHandler()
    color_format = f"%(log_color)s{format._fmt}%(reset)s"
    stream_handler.setFormatter(ColoredFormatter(color_format))

    root_logger.addHandler(stream_handler)