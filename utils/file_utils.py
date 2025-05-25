"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

File handling utilities for SONA AI Assistant.
Handles temporary file creation, cleanup, and file operations.
"""

import os
import tempfile
import hashlib
import shutil
from typing import Optional, Union, Dict, Any
from pathlib import Path
from loguru import logger
import time


def create_temp_file(content: bytes, suffix: str = "", prefix: str = "sona_") -> str:
    """
    Create temporary file with content.

    Args:
        content: File content as bytes
        suffix: File suffix/extension (e.g., '.wav', '.mp3')
        prefix: File prefix for identification

    Returns:
        Path to temporary file

    Raises:
        RuntimeError: If file creation fails
    """
    try:
        # Ensure suffix starts with dot if provided
        if suffix and not suffix.startswith('.'):
            suffix = '.' + suffix

        # Create temporary file with custom prefix and suffix
        temp_fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)

        try:
            # Write content to file
            with os.fdopen(temp_fd, 'wb') as temp_file:
                temp_file.write(content)

            logger.info(f"Created temporary file: {temp_path} ({len(content)} bytes)")
            return temp_path

        except Exception as e:
            # Clean up file descriptor if writing fails
            try:
                os.close(temp_fd)
            except:
                pass
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

    except Exception as e:
        logger.error(f"Failed to create temporary file: {e}")
        raise RuntimeError(f"Temporary file creation failed: {e}")


def create_temp_directory(prefix: str = "sona_dir_") -> str:
    """
    Create temporary directory.

    Args:
        prefix: Directory prefix for identification

    Returns:
        Path to temporary directory

    Raises:
        RuntimeError: If directory creation fails
    """
    try:
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        logger.info(f"Created temporary directory: {temp_dir}")
        return temp_dir

    except Exception as e:
        logger.error(f"Failed to create temporary directory: {e}")
        raise RuntimeError(f"Temporary directory creation failed: {e}")


def cleanup_temp_file(file_path: str) -> bool:
    """
    Clean up temporary file.

    Args:
        file_path: Path to file to delete

    Returns:
        True if successful, False otherwise
    """
    try:
        if not file_path:
            return False

        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
            return True
        else:
            logger.warning(f"File to cleanup does not exist: {file_path}")
            return False

    except Exception as e:
        logger.error(f"Failed to cleanup file {file_path}: {e}")
        return False


def cleanup_temp_directory(dir_path: str) -> bool:
    """
    Clean up temporary directory and all its contents.

    Args:
        dir_path: Path to directory to delete

    Returns:
        True if successful, False otherwise
    """
    try:
        if not dir_path:
            return False

        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
            logger.info(f"Cleaned up temporary directory: {dir_path}")
            return True
        else:
            logger.warning(f"Directory to cleanup does not exist: {dir_path}")
            return False

    except Exception as e:
        logger.error(f"Failed to cleanup directory {dir_path}: {e}")
        return False


def get_file_hash(file_path: str, algorithm: str = "md5") -> Optional[str]:
    """
    Get hash of file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')

    Returns:
        Hash string or None if failed
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found for hashing: {file_path}")
            return None

        # Select hash algorithm
        if algorithm == "md5":
            hash_obj = hashlib.md5()
        elif algorithm == "sha1":
            hash_obj = hashlib.sha1()
        elif algorithm == "sha256":
            hash_obj = hashlib.sha256()
        else:
            logger.error(f"Unsupported hash algorithm: {algorithm}")
            return None

        # Read file in chunks to handle large files
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)

        hash_value = hash_obj.hexdigest()
        logger.debug(f"File hash ({algorithm}): {hash_value} for {file_path}")
        return hash_value

    except Exception as e:
        logger.error(f"Failed to get file hash: {e}")
        return None


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive file information.

    Args:
        file_path: Path to file

    Returns:
        Dictionary with file information
    """
    try:
        if not os.path.exists(file_path):
            return {"exists": False, "error": "File not found"}

        stat = os.stat(file_path)
        path_obj = Path(file_path)

        info = {
            "exists": True,
            "path": str(path_obj.absolute()),
            "name": path_obj.name,
            "stem": path_obj.stem,
            "suffix": path_obj.suffix,
            "size": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "accessed": stat.st_atime,
            "is_file": path_obj.is_file(),
            "is_dir": path_obj.is_dir(),
            "permissions": oct(stat.st_mode)[-3:],
        }

        # Add hash for small files (< 10MB)
        if info["size"] < 10 * 1024 * 1024:
            info["md5"] = get_file_hash(file_path, "md5")

        return info

    except Exception as e:
        logger.error(f"Failed to get file info for {file_path}: {e}")
        return {"exists": False, "error": str(e)}


def ensure_directory_exists(dir_path: str) -> bool:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        dir_path: Path to directory

    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True

    except Exception as e:
        logger.error(f"Failed to create directory {dir_path}: {e}")
        return False


def copy_file(source: str, destination: str, overwrite: bool = False) -> bool:
    """
    Copy file from source to destination.

    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite existing file

    Returns:
        True if successful, False otherwise
    """
    try:
        if not os.path.exists(source):
            logger.error(f"Source file does not exist: {source}")
            return False

        if os.path.exists(destination) and not overwrite:
            logger.error(f"Destination file exists and overwrite is False: {destination}")
            return False

        # Ensure destination directory exists
        dest_dir = os.path.dirname(destination)
        if dest_dir and not ensure_directory_exists(dest_dir):
            return False

        shutil.copy2(source, destination)
        logger.info(f"Copied file: {source} -> {destination}")
        return True

    except Exception as e:
        logger.error(f"Failed to copy file {source} to {destination}: {e}")
        return False


def move_file(source: str, destination: str, overwrite: bool = False) -> bool:
    """
    Move file from source to destination.

    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite existing file

    Returns:
        True if successful, False otherwise
    """
    try:
        if not os.path.exists(source):
            logger.error(f"Source file does not exist: {source}")
            return False

        if os.path.exists(destination) and not overwrite:
            logger.error(f"Destination file exists and overwrite is False: {destination}")
            return False

        # Ensure destination directory exists
        dest_dir = os.path.dirname(destination)
        if dest_dir and not ensure_directory_exists(dest_dir):
            return False

        shutil.move(source, destination)
        logger.info(f"Moved file: {source} -> {destination}")
        return True

    except Exception as e:
        logger.error(f"Failed to move file {source} to {destination}: {e}")
        return False


def read_file_content(file_path: str, encoding: str = 'utf-8') -> Optional[Union[str, bytes]]:
    """
    Read file content safely.

    Args:
        file_path: Path to file
        encoding: Text encoding ('utf-8') or 'binary' for bytes

    Returns:
        File content as string or bytes, None if failed
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None

        if encoding == 'binary':
            with open(file_path, 'rb') as f:
                content = f.read()
        else:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()

        logger.debug(
            f"Read file content: {file_path} ({len(content)} {'bytes' if encoding == 'binary' else 'characters'})")
        return content

    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return None


def write_file_content(file_path: str, content: Union[str, bytes], encoding: str = 'utf-8',
                       overwrite: bool = True) -> bool:
    """
    Write content to file safely.

    Args:
        file_path: Path to file
        content: Content to write
        encoding: Text encoding ('utf-8') or 'binary' for bytes
        overwrite: Whether to overwrite existing file

    Returns:
        True if successful, False otherwise
    """
    try:
        if os.path.exists(file_path) and not overwrite:
            logger.error(f"File exists and overwrite is False: {file_path}")
            return False

        # Ensure directory exists
        dir_path = os.path.dirname(file_path)
        if dir_path and not ensure_directory_exists(dir_path):
            return False

        if isinstance(content, bytes) or encoding == 'binary':
            with open(file_path, 'wb') as f:
                f.write(content if isinstance(content, bytes) else content.encode())
        else:
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)

        logger.info(
            f"Wrote file content: {file_path} ({len(content)} {'bytes' if isinstance(content, bytes) else 'characters'})")
        return True

    except Exception as e:
        logger.error(f"Failed to write file {file_path}: {e}")
        return False


def cleanup_old_temp_files(temp_dir: Optional[str] = None, max_age_hours: int = 24) -> int:
    """
    Clean up old temporary files in the system temp directory.

    Args:
        temp_dir: Temporary directory to clean (default: system temp)
        max_age_hours: Maximum age in hours before cleanup

    Returns:
        Number of files cleaned up
    """
    try:
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()

        if not os.path.exists(temp_dir):
            return 0

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0

        # Look for SONA temporary files
        for filename in os.listdir(temp_dir):
            if filename.startswith("sona_"):
                file_path = os.path.join(temp_dir, filename)

                try:
                    # Check file age
                    file_stat = os.stat(file_path)
                    age_seconds = current_time - file_stat.st_mtime

                    if age_seconds > max_age_seconds:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                            cleaned_count += 1
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                            cleaned_count += 1

                        logger.debug(f"Cleaned up old temp file: {file_path} (age: {age_seconds / 3600:.1f}h)")

                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {file_path}: {e}")
                    continue

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old temporary files")

        return cleaned_count

    except Exception as e:
        logger.error(f"Failed to cleanup old temp files: {e}")
        return 0
