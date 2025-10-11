"""
Structured logging configuration for GuidedPATH
"""

import logging
import sys
from typing import Any, Dict

import structlog


def setup_logging() -> None:
    """
    Configure structured logging for the application.
    """
    # Configure standard library logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer() if settings.DEBUG else JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


class JSONRenderer(structlog.dev.ConsoleRenderer):
    """
    Custom JSON renderer for production logging.
    """

    def __call__(self, logger: Any, name: str, event_dict: Dict[str, Any]) -> str:
        """
        Render log entry as JSON.
        """
        # Add application context
        event_dict.update({
            "service": "guidedpath-backend",
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
        })

        # Remove unnecessary fields for JSON output
        event_dict.pop("logger", None)
        event_dict.pop("level", None)

        return super().__call__(logger, name, event_dict)
