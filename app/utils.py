#Helper functions will be Implemented here

# Helper functions will be implemented here

import logging

def setup_logger(name: str):
    """
    Sets up a logger with the given name.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def chunk_text(text: str, chunk_size: int = 512):
    """
    Splits text into chunks of specified size.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
