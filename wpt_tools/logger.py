"""
Custom logger class to add line numbers to log messages with a level of WARNING or higher.
"""

import logging
import os
from datetime import datetime

import coloredlogs

from wpt_tools.load_env import load_env

load_env()
console_log_level = os.getenv("WPTTOOLS_CONSOLE_LOG_LEVEL", "INFO")
file_log_level = os.getenv("WPTTOOLS_FILE_LOG_LEVEL", "INFO")


class WPTToolsLogger:
    """
    Custom logger class to add line numbers to log messages with a level of WARNING or higher.
    """

    def __init__(
        self,
        console_log_level=console_log_level,
        file_log_level=file_log_level,
        fmt=None,
        log_file=None,
        log_dir="log",
    ):
        """
        Construct the WPTToolsLogger class.

        Parameters
        ----------
        console_log_level : str
            The logging level for the console.
        file_log_level : str
            The logging level for the file.
        fmt : str
            The format for the log messages.
        log_file : str
            The name of the log file.
        log_dir : str
            The directory for the log file.

        """
        if fmt is None:
            fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

        self.console_level = console_log_level
        self.file_level = file_log_level
        self.fmt = fmt
        self.log_dir = log_dir

        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)

        if log_file is None:
            start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file = f"{start_time}.log"

        self.log_file = os.path.join(log_dir, log_file)

    class LineNumberFilter(logging.Filter):
        """
        Custom filter class to add line numbers.
        """

        def filter(self, record) -> bool:
            """
            Add line numbers to log messages with a level of WARNING or higher.

            Parameters
            ----------
            record : LogRecord
                The log record.

            Returns
            -------
            bool
                True if the log message is at least WARNING level, False otherwise.

            """
            if record.levelno >= logging.WARNING:
                record.msg = f"{record.msg} ({record.filename}:{record.lineno})"
            return True

    def get_logger(self, name) -> logging.Logger:
        """
        Create and configure a logger with the provided name.

        Args:
            name (str): The name for the logger, typically use __name__ from the calling module.

        Returns:
            logging.Logger: Configured logger instance.

        """
        logger = logging.getLogger(name)
        logger.setLevel(
            logging.DEBUG
        )  # Set to DEBUG to allow coloredlogs to set the level

        coloredlogs.install(level=self.console_level, logger=logger, fmt=self.fmt)
        line_number_filter = self.LineNumberFilter()
        logger.addFilter(line_number_filter)

        if not any(
            isinstance(handler, logging.StreamHandler) for handler in logger.handlers
        ):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.console_level)
            console_handler.setFormatter(logging.Formatter(self.fmt))
            console_handler.addFilter(line_number_filter)

        if not any(
            isinstance(handler, logging.FileHandler) for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(self.file_level)
            file_handler.setFormatter(logging.Formatter(self.fmt))
            file_handler.addFilter(line_number_filter)
            logger.addHandler(file_handler)

        return logger
