"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

Error handling middleware for SONA AI Assistant.
Provides comprehensive error handling, logging, and user-friendly error responses.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import traceback
import time
from typing import Union, Dict, Any
from loguru import logger

from utils.constants import ERROR_MESSAGES


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Custom middleware for error handling and logging."""

    async def dispatch(self, request: Request, call_next):
        """Process request and handle any errors."""
        start_time = time.time()

        try:
            # Process the request
            response = await call_next(request)

            # Log successful requests
            process_time = time.time() - start_time
            logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )

            return response

        except Exception as e:
            # Log the error with full context
            process_time = time.time() - start_time
            logger.error(
                f"Unhandled error in {request.method} {request.url.path} - "
                f"Time: {process_time:.3f}s - "
                f"Error: {str(e)}\n"
                f"Traceback: {traceback.format_exc()}"
            )

            # Return generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Internal server error occurred",
                    "status_code": 500,
                    "path": str(request.url.path),
                    "method": request.method,
                    "timestamp": time.time(),
                    "request_id": getattr(request.state, 'request_id', None)
                }
            )


def setup_error_handlers(app: FastAPI):
    """Setup global error handlers for the FastAPI application."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with detailed logging and user-friendly responses."""
        logger.warning(
            f"HTTP {exc.status_code} in {request.method} {request.url.path}: {exc.detail}"
        )

        # Map status codes to user-friendly messages
        user_messages = {
            400: "Invalid request. Please check your input and try again.",
            401: "Authentication required. Please provide valid credentials.",
            403: "Access forbidden. You don't have permission to access this resource.",
            404: "Resource not found. The requested endpoint or resource doesn't exist.",
            405: "Method not allowed. This HTTP method is not supported for this endpoint.",
            409: "Conflict. The request conflicts with the current state of the resource.",
            422: "Validation error. Please check your input data.",
            429: "Too many requests. Please slow down and try again later.",
            500: "Internal server error. Please try again later.",
            503: "Service temporarily unavailable. Please try again later."
        }

        user_message = user_messages.get(exc.status_code, exc.detail)

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": user_message,
                "status_code": exc.status_code,
                "path": str(request.url.path),
                "method": request.method,
                "timestamp": time.time(),
                "details": exc.detail if exc.status_code < 500 else None  # Hide internal details for 5xx errors
            }
        )

    @app.exception_handler(StarletteHTTPException)
    async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle Starlette HTTP exceptions."""
        logger.warning(
            f"Starlette HTTP {exc.status_code} in {request.method} {request.url.path}: {exc.detail}"
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.detail or "HTTP error occurred",
                "status_code": exc.status_code,
                "path": str(request.url.path),
                "method": request.method,
                "timestamp": time.time()
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors with detailed field information."""
        logger.warning(f"Validation error in {request.method} {request.url.path}: {exc.errors()}")

        # Format validation errors for user-friendly display
        error_details = []
        for error in exc.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"] if str(loc) != "body")

            # Create user-friendly error messages
            error_msg = error["msg"]
            error_type = error["type"]

            if error_type == "missing":
                user_msg = f"'{field_path}' is required"
            elif error_type == "type_error":
                user_msg = f"'{field_path}' has invalid type"
            elif error_type == "value_error":
                user_msg = f"'{field_path}' has invalid value"
            else:
                user_msg = f"'{field_path}': {error_msg}"

            error_details.append({
                "field": field_path,
                "message": user_msg,
                "type": error_type,
                "input": error.get("input")
            })

        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "Validation failed",
                "status_code": 422,
                "path": str(request.url.path),
                "method": request.method,
                "timestamp": time.time(),
                "validation_errors": error_details
            }
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle ValueError exceptions."""
        logger.error(f"ValueError in {request.method} {request.url.path}: {str(exc)}")

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Invalid input value provided",
                "status_code": 400,
                "path": str(request.url.path),
                "method": request.method,
                "timestamp": time.time(),
                "details": str(exc)
            }
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        """Handle file not found errors."""
        logger.error(f"FileNotFoundError in {request.method} {request.url.path}: {str(exc)}")

        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": "Required file not found",
                "status_code": 404,
                "path": str(request.url.path),
                "method": request.method,
                "timestamp": time.time()
            }
        )

    @app.exception_handler(PermissionError)
    async def permission_error_handler(request: Request, exc: PermissionError):
        """Handle permission errors."""
        logger.error(f"PermissionError in {request.method} {request.url.path}: {str(exc)}")

        return JSONResponse(
            status_code=403,
            content={
                "success": False,
                "error": "Permission denied",
                "status_code": 403,
                "path": str(request.url.path),
                "method": request.method,
                "timestamp": time.time()
            }
        )

    @app.exception_handler(TimeoutError)
    async def timeout_error_handler(request: Request, exc: TimeoutError):
        """Handle timeout errors."""
        logger.error(f"TimeoutError in {request.method} {request.url.path}: {str(exc)}")

        return JSONResponse(
            status_code=408,
            content={
                "success": False,
                "error": "Request timeout. The operation took too long to complete.",
                "status_code": 408,
                "path": str(request.url.path),
                "method": request.method,
                "timestamp": time.time()
            }
        )

    @app.exception_handler(ConnectionError)
    async def connection_error_handler(request: Request, exc: ConnectionError):
        """Handle connection errors (e.g., API service unavailable)."""
        logger.error(f"ConnectionError in {request.method} {request.url.path}: {str(exc)}")

        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "Service temporarily unavailable. Please try again later.",
                "status_code": 503,
                "path": str(request.url.path),
                "method": request.method,
                "timestamp": time.time()
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other unhandled exceptions."""
        # Log the full exception with traceback
        logger.error(
            f"Unhandled exception in {request.method} {request.url.path}: "
            f"{type(exc).__name__}: {str(exc)}\n"
            f"Traceback: {traceback.format_exc()}"
        )

        # Don't expose internal error details to users in production
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "An unexpected error occurred. Please try again later.",
                "status_code": 500,
                "path": str(request.url.path),
                "method": request.method,
                "timestamp": time.time(),
                "exception_type": type(exc).__name__
            }
        )


class APIKeyError(Exception):
    """Custom exception for API key related errors."""
    pass


class ModelNotAvailableError(Exception):
    """Custom exception for when AI model is not available."""
    pass


class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""
    pass


class ImageGenerationError(Exception):
    """Custom exception for image generation errors."""
    pass


class WebSearchError(Exception):
    """Custom exception for web search errors."""
    pass


def setup_custom_exception_handlers(app: FastAPI):
    """Setup handlers for custom exceptions."""

    @app.exception_handler(APIKeyError)
    async def api_key_error_handler(request: Request, exc: APIKeyError):
        """Handle API key errors."""
        logger.error(f"API key error in {request.method} {request.url.path}: {str(exc)}")

        return JSONResponse(
            status_code=401,
            content={
                "success": False,
                "error": "API key error. Please check your configuration.",
                "status_code": 401,
                "path": str(request.url.path),
                "method": request.method,
                "timestamp": time.time()
            }
        )

    @app.exception_handler(ModelNotAvailableError)
    async def model_not_available_handler(request: Request, exc: ModelNotAvailableError):
        """Handle model not available errors."""
        logger.error(f"Model not available in {request.method} {request.url.path}: {str(exc)}")

        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "AI model temporarily unavailable. Please try again later.",
                "status_code": 503,
                "path": str(request.url.path),
                "method": request.method,
                "timestamp": time.time()
            }
        )

    @app.exception_handler(AudioProcessingError)
    async def audio_processing_error_handler(request: Request, exc: AudioProcessingError):
        """Handle audio processing errors."""
        logger.error(f"Audio processing error in {request.method} {request.url.path}: {str(exc)}")

        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "Audio processing failed. Please check your audio file format and try again.",
                "status_code": 422,
                "path": str(request.url.path),
                "method": request.method,
                "timestamp": time.time()
            }
        )

    @app.exception_handler(ImageGenerationError)
    async def image_generation_error_handler(request: Request, exc: ImageGenerationError):
        """Handle image generation errors."""
        logger.error(f"Image generation error in {request.method} {request.url.path}: {str(exc)}")

        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "Image generation failed. Please try a different prompt.",
                "status_code": 422,
                "path": str(request.url.path),
                "method": request.method,
                "timestamp": time.time()
            }
        )

    @app.exception_handler(WebSearchError)
    async def web_search_error_handler(request: Request, exc: WebSearchError):
        """Handle web search errors."""
        logger.error(f"Web search error in {request.method} {request.url.path}: {str(exc)}")

        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "Web search service temporarily unavailable. Please try again later.",
                "status_code": 503,
                "path": str(request.url.path),
                "method": request.method,
                "timestamp": time.time()
            }
        )


def create_error_response(
        status_code: int,
        message: str,
        details: Union[str, Dict[str, Any], None] = None,
        path: str = "",
        method: str = ""
) -> JSONResponse:
    """
    Create standardized error response.

    Args:
        status_code: HTTP status code
        message: Error message
        details: Additional error details
        path: Request path
        method: HTTP method

    Returns:
        JSONResponse with standardized error format
    """
    content = {
        "success": False,
        "error": message,
        "status_code": status_code,
        "timestamp": time.time()
    }

    if path:
        content["path"] = path
    if method:
        content["method"] = method
    if details:
        content["details"] = details

    return JSONResponse(status_code=status_code, content=content)


def log_error_context(request: Request, error: Exception):
    """
    Log error with full request context.

    Args:
        request: FastAPI request object
        error: Exception that occurred
    """
    try:
        # Extract request information
        context = {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client": str(request.client) if request.client else None,
            "error_type": type(error).__name__,
            "error_message": str(error)
        }

        # Log with full context
        logger.error(f"Request error context: {context}")

    except Exception as e:
        logger.error(f"Failed to log error context: {e}")


# Error monitoring and metrics (placeholder for future implementation)
class ErrorMetrics:
    """Track error metrics for monitoring."""

    def __init__(self):
        self.error_counts = {}
        self.error_rates = {}

    def record_error(self, error_type: str, endpoint: str):
        """Record an error occurrence."""
        key = f"{error_type}:{endpoint}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error metrics."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_breakdown": self.error_counts,
            "top_errors": sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


# Global error metrics instance
error_metrics = ErrorMetrics()
