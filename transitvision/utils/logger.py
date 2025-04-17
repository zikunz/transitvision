"""Logging utility for the TransitVision package."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any


def setup_logger(
    name: str = "transitvision",
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
    log_format: Optional[str] = None,
    propagate: bool = False,
) -> logging.Logger:
    """Set up and configure a logger.
    
    Args:
        name: Name of the logger.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to a log file.
        console: Whether to log to console.
        log_format: Custom log format string.
        propagate: Whether to propagate logs to parent loggers.
        
    Returns:
        Configured logger instance.
    """
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Set level
    logger.setLevel(level)
    
    # Set propagation
    logger.propagate = propagate
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log file specified
    if log_file:
        # Create directory if it doesn't exist
        log_path = Path(log_file)
        if not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "transitvision") -> logging.Logger:
    """Get an existing logger or create a new one with default settings.
    
    Args:
        name: Name of the logger.
        
    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set up a new one
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger


class LogManager:
    """Centralized manager for logging configuration.
    
    This class provides methods for managing loggers across the package,
    allowing consistent configuration of multiple loggers.
    """
    
    def __init__(self) -> None:
        """Initialize the log manager."""
        self.loggers: Dict[str, logging.Logger] = {}
        self.default_level = logging.INFO
        self.default_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.log_file = None
        self.console = True
    
    def configure(
        self,
        level: int = logging.INFO,
        log_format: Optional[str] = None,
        log_file: Optional[Union[str, Path]] = None,
        console: bool = True,
    ) -> None:
        """Configure default settings for all loggers.
        
        Args:
            level: Logging level.
            log_format: Log message format.
            log_file: Path to log file.
            console: Whether to log to console.
        """
        self.default_level = level
        
        if log_format is not None:
            self.default_format = log_format
        
        self.log_file = log_file
        self.console = console
        
        # Update existing loggers
        for name, logger in self.loggers.items():
            self._configure_logger(logger)
    
    def _configure_logger(self, logger: logging.Logger) -> None:
        """Apply current configuration to a logger.
        
        Args:
            logger: Logger to configure.
        """
        # Clear existing handlers
        if logger.handlers:
            logger.handlers.clear()
        
        # Set level
        logger.setLevel(self.default_level)
        
        # Create formatter
        formatter = logging.Formatter(self.default_format)
        
        # Add console handler if enabled
        if self.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Add file handler if log file specified
        if self.log_file:
            # Create directory if it doesn't exist
            log_path = Path(self.log_file)
            if not log_path.parent.exists():
                log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    def get_logger(self, name: str = "transitvision") -> logging.Logger:
        """Get or create a logger with current settings.
        
        Args:
            name: Name of the logger.
            
        Returns:
            Configured logger instance.
        """
        if name in self.loggers:
            return self.loggers[name]
        
        # Create new logger
        logger = logging.getLogger(name)
        
        # Configure it
        self._configure_logger(logger)
        
        # Store for future reference
        self.loggers[name] = logger
        
        return logger
    
    def set_level(self, level: int, logger_name: Optional[str] = None) -> None:
        """Set logging level for one or all loggers.
        
        Args:
            level: Logging level.
            logger_name: Optional name of specific logger to configure.
        """
        if logger_name is None:
            # Update default level
            self.default_level = level
            
            # Update all existing loggers
            for logger in self.loggers.values():
                logger.setLevel(level)
        elif logger_name in self.loggers:
            # Update specific logger
            self.loggers[logger_name].setLevel(level)
    
    def add_file_handler(
        self,
        log_file: Union[str, Path],
        logger_name: Optional[str] = None,
        level: Optional[int] = None,
    ) -> None:
        """Add a file handler to one or all loggers.
        
        Args:
            log_file: Path to log file.
            logger_name: Optional name of specific logger to configure.
            level: Optional specific level for this handler.
        """
        # Create formatter
        formatter = logging.Formatter(self.default_format)
        
        # Create directory if it doesn't exist
        log_path = Path(log_file)
        if not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        
        # Set level if specified
        if level is not None:
            file_handler.setLevel(level)
        
        if logger_name is None:
            # Add to all loggers
            for logger in self.loggers.values():
                logger.addHandler(file_handler)
        elif logger_name in self.loggers:
            # Add to specific logger
            self.loggers[logger_name].addHandler(file_handler)


# Create a global log manager instance
log_manager = LogManager()


# Convenience function to get a logger using the global manager
def get_managed_logger(name: str = "transitvision") -> logging.Logger:
    """Get a logger managed by the global LogManager.
    
    Args:
        name: Name of the logger.
        
    Returns:
        Configured logger instance.
    """
    return log_manager.get_logger(name)