"""
utils.py
--------
Shared utility functions: logging setup, timing decorators, and general helpers.
"""

import functools
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from src.config import LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a consistently configured logger.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__`` of the calling module).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    return logger


def timer(func: Callable) -> Callable:
    """
    Decorator that logs execution time for any function.

    Parameters
    ----------
    func : Callable
        Function to wrap.

    Returns
    -------
    Callable
        Wrapped function.
    """
    logger = get_logger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%-40s completed in %.3fs", func.__qualname__, elapsed)
        return result

    return wrapper


def ensure_dir(path: Path) -> Path:
    """
    Create directory (and parents) if it does not already exist.

    Parameters
    ----------
    path : Path
        Directory path to create.

    Returns
    -------
    Path
        The (now existing) directory path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: Path) -> None:
    """
    Persist a dictionary as a JSON file.

    Parameters
    ----------
    data : dict
        Data to serialise.
    path : Path
        Destination file path.
    """
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_json(path: Path) -> dict:
    """
    Load a JSON file into a dictionary.

    Parameters
    ----------
    path : Path
        Source file path.

    Returns
    -------
    dict
        Parsed JSON content.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
