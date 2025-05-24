#!/usr/bin/env python3
"""
SONA AI Assistant - Main Entry Point
Handles application startup and orchestrates backend and frontend services.
"""

import sys
import os
import asyncio
import argparse
from typing import Optional
import multiprocessing
import subprocess
import time
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import get_settings
from backend.app import sona_backend
from utils.validation import validate_api_keys, validate_model_configuration
from utils.constants import SONA_PERSONA


def setup_environment():
    """Setup environment and validate configuration."""
    try:
        settings = get_settings()

        logger.info(f"Starting {SONA_PERSONA['name']} v{settings.app_version}")
        logger.info(f"Debug mode: {settings.debug}")

        # Validate API keys
        api_validation = validate_api_keys()
        missing_keys = [service for service, valid in api_validation.items() if not valid]

        if missing_keys:
            logger.warning(f"Missing API keys for: {', '.join(missing_keys)}")
            logger.warning("Some features may not be available.")

        # Validate model configuration
        model_validation = validate_model_configuration()
        invalid_models = [model for model, valid in model_validation.items() if not valid]

        if invalid_models:
            logger.error(f"Invalid model configuration for: {', '.join(invalid_models)}")
            return False

        logger.info("Environment validation completed successfully")
        return True

    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        return False


def run_backend():
    """Run the FastAPI backend server."""
    try:
        logger.info("Starting SONA backend server...")
        sona_backend.run()
    except Exception as e:
        logger.error(f"Backend server failed: {e}")
        sys.exit(1)


def run_frontend():
    """Run the Streamlit frontend."""
    try:
        logger.info("Starting SONA frontend server...")

        # Wait for backend to be ready
        time.sleep(2)

        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "ui/streamlit_app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--browser.serverAddress=localhost",
            "--browser.gatherUsageStats=false"
        ])

    except Exception as e:
        logger.error(f"Frontend server failed: {e}")
        sys.exit(1)


def run_development_mode():
    """Run in development mode with both backend and frontend."""
    try:
        logger.info("Starting SONA in development mode...")

        # Start backend in a separate process
        backend_process = multiprocessing.Process(target=run_backend)
        backend_process.start()

        # Give backend time to start
        time.sleep(3)

        # Start frontend in main process
        run_frontend()

    except KeyboardInterrupt:
        logger.info("Shutting down SONA...")
        if 'backend_process' in locals():
            backend_process.terminate()
            backend_process.join()
    except Exception as e:
        logger.error(f"Development mode failed: {e}")
        if 'backend_process' in locals():
            backend_process.terminate()
            backend_process.join()
        sys.exit(1)


def run_production_mode():
    """Run in production mode (backend only)."""
    try:
        logger.info("Starting SONA in production mode (backend only)...")
        run_backend()
    except Exception as e:
        logger.error(f"Production mode failed: {e}")
        sys.exit(1)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="SONA AI Assistant")
    parser.add_argument(
        "--mode",
        choices=["dev", "backend", "frontend"],
        default="dev",
        help="Run mode: dev (both), backend (API only), frontend (UI only)"
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind to (overrides config)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (overrides config)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    # Setup environment
    if not setup_environment():
        logger.error("Environment setup failed. Exiting.")
        sys.exit(1)

    # Override settings if specified
    settings = get_settings()
    if args.debug:
        settings.debug = True
    if args.host:
        settings.backend_host = args.host
    if args.port:
        settings.backend_port = args.port

    # Run based on mode
    try:
        if args.mode == "dev":
            run_development_mode()
        elif args.mode == "backend":
            run_production_mode()
        elif args.mode == "frontend":
            run_frontend()
    except KeyboardInterrupt:
        logger.info("SONA shutting down gracefully...")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
