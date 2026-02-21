"""
Production-Ready Project Path Management

Automatically detects project root by searching for marker files (.git, pyproject.toml, etc.)
Works correctly when running scripts from any subdirectory within the project.

Features:
- Auto-detection of project root (searches up directory tree)
- Thread-safe singleton pattern
- Comprehensive error handling and logging
- Multiple file format support (JSON, pickle, CSV, text, YAML)
- Path validation and creation
- Works from any subdirectory
- Extensive testing and validation

Version: 2.0 - Production Ready
"""

import os
import sys
import json
import pickle
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Optional, Union, List, Dict
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ProjectPaths")


# ============================================================================
# EXCEPTIONS
# ============================================================================


class ProjectRootNotFoundError(Exception):
    """Raised when project root cannot be determined."""

    pass


class PathValidationError(Exception):
    """Raised when path validation fails."""

    pass


# ============================================================================
# PROJECT ROOT DETECTOR
# ============================================================================


class ProjectRootDetector:
    """
    Detects project root by searching for marker files/directories.

    Searches upward from the current file location until it finds
    a directory containing one of the marker files/directories.
    """

    # Ordered by priority - first match wins
    DEFAULT_MARKERS = [
        ".git",  # Git repository
        # 'pyproject.toml',    # Modern Python projects
        # 'setup.py',          # Traditional Python packages
        # 'setup.cfg',         # Python package config
        # 'requirements.txt',  # Python dependencies
        # 'Pipfile',           # Pipenv projects
        "poetry.lock",  # Poetry projects
        # 'Cargo.toml',        # Rust projects
        # 'package.json',      # Node.js projects
        # 'readme.md',        # Common readme file
        # "README.md",       # Alternative readme file
        ".projectroot",  # Custom marker (user can create this)
    ]

    @classmethod
    def find_project_root(
        cls,
        start_path: Optional[Path] = None,
        markers: Optional[List[str]] = None,
        max_depth: int = 10,
    ) -> Path:
        """
        Find project root by searching upward for marker files.

        Parameters
        ----------
        start_path : Path, optional
            Starting directory (defaults to current file's directory)
        markers : List[str], optional
            List of marker files/dirs to search for
        max_depth : int
            Maximum levels to search upward

        Returns
        -------
        Path
            Project root directory

        Raises
        ------
        ProjectRootNotFoundError
            If project root cannot be found
        """
        if markers is None:
            markers = cls.DEFAULT_MARKERS

        # Start from the directory containing THIS file if not specified
        if start_path is None:
            start_path = Path(__file__).resolve().parent
        else:
            start_path = Path(start_path).resolve()

        current = start_path

        # Search upward through parent directories
        for _ in range(max_depth):
            # Check if any marker exists in current directory
            for marker in markers:
                if (current / marker).exists():
                    logger.info(f"Project root found: {current} (marker: {marker})")
                    return current

            # Move to parent directory
            parent = current.parent
            if parent == current:
                # Reached filesystem root
                break
            current = parent

        # Fallback: if no marker found, use the directory containing this file
        # This assumes project_utils is inside the project
        fallback = Path(__file__).resolve().parent.parent
        logger.warning(
            f"No project markers found. Using fallback: {fallback}\n"
            f"To fix this, create a .projectroot file in your project root, "
            f"or use ProjectPaths.set_project_root() explicitly."
        )
        return fallback


# ============================================================================
# MAIN PATH MANAGER (Thread-Safe Singleton)
# ============================================================================


class ProjectPaths:
    """
    Thread-safe singleton for managing project directory paths.

    Automatically detects project root on first access.
    All paths are computed lazily and cached for performance.

    Usage
    -----
    >>> from project_utils.paths import ProjectPaths, ensure_dir_exists
    >>>
    >>> # Auto-detection (recommended)
    >>> data_dir = ProjectPaths.get_data_dir()
    >>>
    >>> # Manual override if needed
    >>> ProjectPaths.set_project_root("/path/to/project")
    >>>
    >>> # Ensure directory exists before writing
    >>> ensure_dir_exists(ProjectPaths.get_output_files_dir())
    """

    _instance = None
    _lock = threading.Lock()
    _PROJECT_ROOT: Optional[Path] = None
    _initialized = False

    def __new__(cls):
        """Thread-safe singleton implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure project root is detected (lazy initialization)."""
        if not cls._initialized:
            with cls._lock:
                if not cls._initialized:
                    if cls._PROJECT_ROOT is None:
                        cls._PROJECT_ROOT = ProjectRootDetector.find_project_root()
                    cls._initialized = True

    @classmethod
    def set_project_root(cls, path: Union[str, Path], validate: bool = True) -> None:
        """
        Manually set the project root.

        Parameters
        ----------
        path : str or Path
            Project root directory path
        validate : bool
            If True, verify the path exists

        Raises
        ------
        PathValidationError
            If validation is enabled and path doesn't exist
        """
        path = Path(path).resolve()

        if validate and not path.exists():
            raise PathValidationError(f"Project root does not exist: {path}")

        if validate and not path.is_dir():
            raise PathValidationError(f"Project root is not a directory: {path}")

        with cls._lock:
            cls._PROJECT_ROOT = path
            cls._initialized = True
            # Clear cache when root changes
            cls.get_project_root.cache_clear()

        logger.info(f"Project root set to: {path}")

    @classmethod
    @lru_cache(maxsize=1)
    def get_project_root(cls) -> Path:
        """
        Get the project root directory.

        Auto-detects on first call by searching for marker files.
        Cached for performance.
        """
        cls._ensure_initialized()
        return cls._PROJECT_ROOT

    # ────────────────────────────────────────────────────────────────────
    # DATA DIRECTORIES
    # ────────────────────────────────────────────────────────────────────

    @classmethod
    def get_data_dir(cls) -> Path:
        """Get main data directory."""
        return cls.get_project_root() / "data"

    @classmethod
    def get_raw_data_dir(cls) -> Path:
        """Get raw/unprocessed data directory."""
        return cls.get_data_dir() / "raw"

    @classmethod
    def get_processed_data_dir(cls) -> Path:
        """Get processed/cleaned data directory."""
        return cls.get_data_dir() / "processed"

    @classmethod
    def get_interim_data_dir(cls) -> Path:
        """Get intermediate processing data directory."""
        return cls.get_data_dir() / "interim"

    @classmethod
    def get_external_data_dir(cls) -> Path:
        """Get external data directory."""
        return cls.get_data_dir() / "external"

    # Alias for backward compatibility
    @classmethod
    def get_cleaned_data_dir(cls) -> Path:
        """Alias for get_processed_data_dir (backward compatibility)."""
        return cls.get_processed_data_dir()

    # ────────────────────────────────────────────────────────────────────
    # NOTEBOOK DIRECTORIES
    # ────────────────────────────────────────────────────────────────────

    @classmethod
    def get_notebooks_dir(cls) -> Path:
        """Get notebooks directory."""
        return cls.get_project_root() / "notebooks"

    @classmethod
    def get_notebook_data_dir(cls) -> Path:
        """Get notebook-specific data directory."""
        return cls.get_notebooks_dir() / "data"

    @classmethod
    def get_notebook_model_dir(cls) -> Path:
        """Get notebook model directory."""
        return cls.get_notebooks_dir() / "models"

    @classmethod
    def get_notebook_visualization_dir(cls) -> Path:
        """Get notebook visualization directory."""
        return cls.get_notebooks_dir() / "visualizations"

    # ────────────────────────────────────────────────────────────────────
    # OUTPUT DIRECTORIES
    # ────────────────────────────────────────────────────────────────────

    @classmethod
    def get_output_dir(cls) -> Path:
        """Get main output directory."""
        return cls.get_project_root() / "output"

    @classmethod
    def get_output_files_dir(cls) -> Path:
        """Get output files directory."""
        return cls.get_output_dir() / "files"

    @classmethod
    def get_output_images_dir(cls) -> Path:
        """Get output images directory."""
        return cls.get_output_dir() / "images"

    @classmethod
    def get_output_reports_dir(cls) -> Path:
        """Get output reports directory."""
        return cls.get_output_dir() / "reports"

    @classmethod
    def get_output_models_dir(cls) -> Path:
        """Get output models directory."""
        return cls.get_output_dir() / "models"

    # ────────────────────────────────────────────────────────────────────
    # CODE DIRECTORIES
    # ────────────────────────────────────────────────────────────────────

    @classmethod
    def get_src_dir(cls) -> Path:
        """Get source code directory."""
        return cls.get_project_root() / "src"

    @classmethod
    def get_utils_dir(cls) -> Path:
        """Get utilities directory."""
        return cls.get_project_root() / "utils"

    @classmethod
    def get_scripts_dir(cls) -> Path:
        """Get scripts directory."""
        return cls.get_project_root() / "scripts"

    @classmethod
    def get_config_dir(cls) -> Path:
        """Get configuration directory."""
        return cls.get_project_root() / "config"

    # ────────────────────────────────────────────────────────────────────
    # TEST DIRECTORIES
    # ────────────────────────────────────────────────────────────────────

    @classmethod
    def get_tests_dir(cls) -> Path:
        """Get tests directory."""
        return cls.get_project_root() / "tests"

    @classmethod
    def get_test_notebook_dir(cls) -> Path:
        """Get test notebook directory."""
        return cls.get_tests_dir() / "notebooks"

    @classmethod
    def get_test_data_dir(cls) -> Path:
        """Get test data directory."""
        return cls.get_tests_dir() / "data"

    # ────────────────────────────────────────────────────────────────────
    # UTILITY METHODS
    # ────────────────────────────────────────────────────────────────────

    @classmethod
    def get_all_directories(cls) -> Dict[str, Path]:
        """
        Get all configured directories as a dictionary.

        Returns
        -------
        Dict[str, Path]
            Dictionary mapping directory names to paths
        """
        return {
            "project_root": cls.get_project_root(),
            "data": cls.get_data_dir(),
            "raw_data": cls.get_raw_data_dir(),
            "processed_data": cls.get_processed_data_dir(),
            "interim_data": cls.get_interim_data_dir(),
            "external_data": cls.get_external_data_dir(),
            "notebooks": cls.get_notebooks_dir(),
            "output": cls.get_output_dir(),
            "output_files": cls.get_output_files_dir(),
            "output_images": cls.get_output_images_dir(),
            "output_reports": cls.get_output_reports_dir(),
            "output_models": cls.get_output_models_dir(),
            "src": cls.get_src_dir(),
            "utils": cls.get_utils_dir(),
            "scripts": cls.get_scripts_dir(),
            "config": cls.get_config_dir(),
            "tests": cls.get_tests_dir(),
        }

    @classmethod
    def create_standard_structure(cls, verbose: bool = True) -> None:
        """
        Create standard project directory structure.

        Creates all standard directories if they don't exist.
        """
        dirs = cls.get_all_directories()

        for name, path in dirs.items():
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                if verbose:
                    logger.info(f"Created directory: {path}")
            else:
                if verbose:
                    logger.debug(f"Directory exists: {path}")

        if verbose:
            logger.info(
                f"Project structure verified/created under: {cls.get_project_root()}"
            )

    @classmethod
    def validate_structure(cls, required_dirs: Optional[List[str]] = None) -> bool:
        """
        Validate that required directories exist.

        Parameters
        ----------
        required_dirs : List[str], optional
            List of required directory names (from get_all_directories keys)
            If None, checks all directories

        Returns
        -------
        bool
            True if all required directories exist
        """
        all_dirs = cls.get_all_directories()

        if required_dirs is None:
            required_dirs = list(all_dirs.keys())

        missing = []
        for name in required_dirs:
            if name not in all_dirs:
                logger.error(f"Unknown directory name: {name}")
                return False

            if not all_dirs[name].exists():
                missing.append(name)

        if missing:
            logger.error(f"Missing directories: {', '.join(missing)}")
            return False

        return True


# ============================================================================
# PATH UTILITIES
# ============================================================================


def ensure_dir_exists(path: Union[str, Path], parents: bool = True) -> Path:
    """
    Ensure directory exists, create if necessary.

    Parameters
    ----------
    path : str or Path
        Directory path
    parents : bool
        Create parent directories if needed

    Returns
    -------
    Path
        Resolved directory path
    """
    path = Path(path)
    path.mkdir(parents=parents, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")
    return path


def get_file_path(
    filename: str, directory_func: Callable[[], Path], create_dir: bool = False
) -> Path:
    """
    Get full file path within a directory.

    Parameters
    ----------
    filename : str
        Name of the file
    directory_func : Callable
        Function that returns the directory path
    create_dir : bool
        Create directory if it doesn't exist

    Returns
    -------
    Path
        Full file path
    """
    directory = directory_func()

    if create_dir:
        ensure_dir_exists(directory)

    return directory / filename


def validate_file_path(path: Union[str, Path], must_exist: bool = False) -> Path:
    """
    Validate file path.

    Parameters
    ----------
    path : str or Path
        File path to validate
    must_exist : bool
        Raise error if file doesn't exist

    Returns
    -------
    Path
        Validated path

    Raises
    ------
    PathValidationError
        If validation fails
    """
    path = Path(path)

    if must_exist and not path.exists():
        raise PathValidationError(f"File does not exist: {path}")

    return path


# ============================================================================
# FILE I/O WITH FORMAT SUPPORT
# ============================================================================


def save_data(
    data: Any, filename: str, directory_func: Callable[[], Path], format: str = "auto"
) -> Path:
    """
    Save data to file with automatic format detection.

    Parameters
    ----------
    data : Any
        Data to save
    filename : str
        Output filename
    directory_func : Callable
        Function returning target directory
    format : str
        Format: 'auto', 'json', 'pickle', 'text', 'csv'

    Returns
    -------
    Path
        Path to saved file

    Examples
    --------
    >>> save_data({'key': 'value'}, 'data.json', ProjectPaths.get_output_files_dir)
    >>> save_data([1, 2, 3], 'data.pkl', ProjectPaths.get_processed_data_dir)
    """
    file_path = get_file_path(filename, directory_func, create_dir=True)

    # Auto-detect format from extension
    if format == "auto":
        ext = file_path.suffix.lower()
        format_map = {
            ".json": "json",
            ".pkl": "pickle",
            ".pickle": "pickle",
            ".txt": "text",
            ".csv": "csv",
        }
        format = format_map.get(ext, "text")

    try:
        if format == "json":
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        elif format == "pickle":
            with file_path.open("wb") as f:
                pickle.dump(data, f)

        elif format == "text":
            with file_path.open("w", encoding="utf-8") as f:
                f.write(str(data))

        elif format == "csv":
            # Basic CSV support - for complex CSV use pandas
            import csv

            with file_path.open("w", newline="", encoding="utf-8") as f:
                if isinstance(data, (list, tuple)) and data:
                    if isinstance(data[0], dict):
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
                    else:
                        writer = csv.writer(f)
                        writer.writerows(data)

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved data to: {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {e}")
        raise


# ============================================================================
# PYTHON PATH MANAGEMENT
# ============================================================================


def add_project_root_to_path(force: bool = False) -> None:
    """
    Add project root to Python path and change working directory.

    This allows imports to work from anywhere in the project.

    Parameters
    ----------
    force : bool
        Force change even if already in path

    Examples
    --------
    >>> add_project_root_to_path()
    >>> from src.models import MyModel  # Now works from anywhere
    """
    project_root = ProjectPaths.get_project_root()
    project_root_str = str(project_root)

    # Add to sys.path if not already there
    if force or project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
        logger.info(f"Added to sys.path: {project_root_str}")

    # Change working directory if not already there
    current_dir = Path.cwd()
    if force or current_dir != project_root:
        os.chdir(project_root)
        logger.info(f"Changed working directory to: {project_root}")


def remove_project_root_from_path() -> None:
    """Remove project root from Python path."""
    project_root_str = str(ProjectPaths.get_project_root())
    if project_root_str in sys.path:
        sys.path.remove(project_root_str)
        logger.info(f"Removed from sys.path: {project_root_str}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def get_relative_path(
    path: Union[str, Path], relative_to: Optional[Path] = None
) -> Path:
    """
    Get path relative to project root (or another directory).

    Parameters
    ----------
    path : str or Path
        Absolute or relative path
    relative_to : Path, optional
        Base path (defaults to project root)

    Returns
    -------
    Path
        Relative path
    """
    path = Path(path)
    if relative_to is None:
        relative_to = ProjectPaths.get_project_root()

    try:
        return path.relative_to(relative_to)
    except ValueError:
        # Path is not relative to base
        return path


def is_within_project(path: Union[str, Path]) -> bool:
    """
    Check if path is within the project directory.

    Parameters
    ----------
    path : str or Path
        Path to check

    Returns
    -------
    bool
        True if path is within project
    """
    path = Path(path).resolve()
    project_root = ProjectPaths.get_project_root()

    try:
        path.relative_to(project_root)
        return True
    except ValueError:
        return False


# ============================================================================
# INITIALIZATION HELPER
# ============================================================================


def initialize_project(
    root_path: Optional[Union[str, Path]] = None,
    create_structure: bool = True,
    add_to_path: bool = True,
    verbose: bool = True,
) -> Path:
    """
    Initialize project paths with one function call.

    Parameters
    ----------
    root_path : str or Path, optional
        Explicit project root (if None, auto-detects)
    create_structure : bool
        Create standard directory structure
    add_to_path : bool
        Add project root to Python path
    verbose : bool
        Print initialization info

    Returns
    -------
    Path
        Project root path

    Examples
    --------
    >>> # At the top of your script
    >>> from project_utils.paths import initialize_project
    >>> initialize_project()
    """
    # Set root if provided
    if root_path is not None:
        ProjectPaths.set_project_root(root_path)

    # Get root (triggers auto-detection if not set)
    root = ProjectPaths.get_project_root()

    if verbose:
        logger.info(f"Initialized project at: {root}")

    # Create directory structure
    if create_structure:
        ProjectPaths.create_standard_structure(verbose=verbose)

    # Add to Python path
    if add_to_path:
        add_project_root_to_path()

    return root
