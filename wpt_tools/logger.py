"""
Logger functions.
"""

import logging

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
    The format of each logging message: {timestamp} : {level name} : {filename} - {message}

    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename)
    format = logging.Formatter('%(asctime)s : %(levelname)s : %(filename)s - %(message)s')
    file_handler.setFormatter(format)

    if root_logger.hasHandlers():
           root_logger.handlers.clear()
   
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(format)
    root_logger.addHandler(stream_handler)