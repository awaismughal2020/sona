# backend/middleware/logger.py
"""
Logging middleware and configuration for SONA AI Assistant.
"""
import sys
from loguru import logger
from config.settings import get_settings
from utils.constants import LOG_FORMAT


def setup_logging():
    """Setup application logging configuration."""
    settings = get_settings()

    # Remove default logger
    logger.remove()

    # Add console logger
    logger.add(
        sys.stdout,
        format=LOG_FORMAT,
        level=settings.log_level,
        colorize=True,
        backtrace=settings.debug,
        diagnose=settings.debug
    )

    # Add file logger if not in debug mode
    if not settings.debug:
        logger.add(
            "logs/sona.log",
            format=LOG_FORMAT,
            level=settings.log_level,
            rotation="1 day",
            retention="1 week",
            compression="zip"
        )

    logger.info("Logging system initialized")


# backend/middleware/error_handler.py
"""
Error handling middleware for SONA AI Assistant.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback
from loguru import logger

from utils.constants import ERROR_MESSAGES


def setup_error_handlers(app: FastAPI):
    """Setup global error handlers for the FastAPI application."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url)
            }
        )

    @app.exception_handler(StarletteHTTPException)
    async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle Starlette HTTP exceptions."""
        logger.warning(f"Starlette HTTP exception: {exc.status_code} - {exc.detail}")

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.detail or "HTTP error occurred",
                "status_code": exc.status_code,
                "path": str(request.url)
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        logger.warning(f"Validation error: {exc.errors()}")

        # Format validation errors
        error_details = []
        for error in exc.errors():
            error_details.append({
                "field": " -> ".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })

        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "Validation failed",
                "details": error_details,
                "status_code": 422,
                "path": str(request.url)
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions."""
        logger.error(f"Unhandled exception: {type(exc).__name__}: {str(exc)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error occurred",
                "status_code": 500,
                "path": str(request.url),
                "exception_type": type(exc).__name__
            }
        )
