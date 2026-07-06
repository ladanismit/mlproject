"""Custom exception handling module for detailed traceback extraction.

This module defines standard exception handling structures for enterprise applications,
capturing script name, line number, and original error message to ease debugging.
"""

import sys
from types import ModuleType


def error_message_detail(error: Exception, error_detail: ModuleType) -> str:
    """Generates a detailed error message containing script name, line number,
    and original exception details.
    
    Args:
        error (Exception): The encountered exception.
        error_detail (ModuleType): The sys module to retrieve execution frame.
        
    Returns:
        str: Formatted error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    
    file_name: str = "Unknown"
    line_number: int = 0
    
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        
    return (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{line_number}] "
        f"error message [{str(error)}]"
    )


class CustomException(Exception):
    """Custom exception class to structure and format detailed error reporting.
    
    Inherits from the base Exception class.
    """
    
    def __init__(self, error_message: Exception, error_detail: ModuleType) -> None:
        """Initializes the CustomException with detailed traceback details.
        
        Args:
            error_message (Exception): The original exception.
            error_detail (ModuleType): The sys module.
        """
        super().__init__(str(error_message))
        self.error_message: str = error_message_detail(
            error_message, error_detail=error_detail
        )
        
    def __str__(self) -> str:
        """Returns the detailed string representation of the exception."""
        return self.error_message
