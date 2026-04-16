"""
Logging configuration for psychometric analysis feature.

Provides centralized logging setup for error tracking and reporting
during psychometric analysis operations.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "psychometric_analysis",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure a logger for psychometric analysis.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional file path for logging output
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_subject_error(
    logger: logging.Logger,
    subject_id: str,
    operation: str,
    error_message: str
) -> None:
    """
    Log an error with subject and operation context.
    
    Args:
        logger: Logger instance
        subject_id: Subject identifier
        operation: Operation name (e.g., 'gausFit', 'plot_psychometric')
        error_message: Error message
    """
    logger.error(f"[{subject_id}] [{operation}] {error_message}")


def log_subject_info(
    logger: logging.Logger,
    subject_id: str,
    message: str
) -> None:
    """
    Log an info message with subject context.
    
    Args:
        logger: Logger instance
        subject_id: Subject identifier
        message: Info message
    """
    logger.info(f"[{subject_id}] {message}")
