import os
import shutil
from pathlib import Path
from typing import Union, List, Optional


def dir_maker(full_path: str) -> None:
    """Create new directory from the given path with proper error handling.

    Parameters
    ----------
    full_path : str
        Absolute path of the new directory.

    Raises
    ----
    OSError
        If directory creation fails due to permissions or invalid path
    ValueError
        If full_path is empty or None
    """
    if not full_path:
        raise ValueError("Directory path cannot be empty or None")

    try:
        # Clean the path to handle any path inconsistencies
        clean_path = os.path.normpath(full_path)
        os.makedirs(clean_path, exist_ok=True)
        print(f"Directory created/verified: {clean_path}")
    except OSError as e:
        raise OSError(f"Failed to create directory {full_path}: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error creating directory {full_path}: {str(e)}")


def dir_path_generator(cur_dir: str, backstep_num: int) -> str:
    """Generate path for the main project directory by stepping back from current directory.

    Parameters
    ----------
    cur_dir : str
        Current directory path
    backstep_num : int
        Number of steps back to take to reach the main directory.
        Must be a positive integer.

    Returns
    -------
    str
        Main directory path

    Raises
    ------
    ValueError
        If backstep_num is negative or cur_dir is invalid
    OSError
        If the resulting path doesn't exist
    """
    if backstep_num < 0:
        raise ValueError("backstep_num must be a non-negative integer")

    if not cur_dir or not os.path.exists(cur_dir):
        raise ValueError(f"Current directory path is invalid: {cur_dir}")

    try:
        no_of_steps = [".."] * backstep_num
        main_dir = os.path.abspath(os.path.join(cur_dir, *no_of_steps))

        # Normalize the path for consistency
        main_dir = os.path.normpath(main_dir)

        if not os.path.exists(main_dir):
            raise OSError(f"Generated path does not exist: {main_dir}")

        return main_dir
    except Exception as e:
        raise Exception(f"Error generating directory path: {str(e)}")


def ensure_dir_exists(directory_path: str) -> bool:
    """Check if directory exists and create it if it doesn't.

    Parameters
    ----------
    directory_path : str
        Path to directory to check/create

    Returns
    -------
    bool
        True if directory exists or was created successfully
    """
    try:
        if not os.path.exists(directory_path):
            dir_maker(directory_path)
            return True
        return True
    except Exception:
        return False


def safe_remove_dir(directory_path: str, force: bool = False) -> bool:
    """Safely remove a directory and all its contents.

    Parameters
    ----------
    directory_path : str
        Path to directory to remove
    force : bool, optional
        If True, remove even if directory is not empty, by default False

    Returns
    -------
    bool
        True if directory was removed successfully
    """
    try:
        if not os.path.exists(directory_path):
            print(f"Directory does not exist: {directory_path}")
            return True

        if force:
            shutil.rmtree(directory_path)
            print(f"Directory forcefully removed: {directory_path}")
        else:
            # Only remove if empty
            os.rmdir(directory_path)
            print(f"Empty directory removed: {directory_path}")
        return True
    except OSError as e:
        print(f"Error removing directory {directory_path}: {str(e)}")
        return False


def list_subdirectories(parent_dir: str) -> List[str]:
    """List all subdirectories in a given directory.

    Parameters
    ----------
    parent_dir : str
        Path to parent directory

    Returns
    -------
    List[str]
        List of subdirectory paths
    """
    try:
        if not os.path.exists(parent_dir):
            return []

        subdirs = []
        for item in os.listdir(parent_dir):
            item_path = os.path.join(parent_dir, item)
            if os.path.isdir(item_path):
                subdirs.append(item_path)
        return subdirs
    except Exception:
        return []


def get_directory_size(directory_path: str) -> int:
    """Get the total size of a directory in bytes.

    Parameters
    ----------
    directory_path : str
        Path to directory

    Returns
    -------
    int
        Size in bytes, 0 if directory doesn't exist or on error
    """
    try:
        if not os.path.exists(directory_path):
            return 0

        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    continue
        return total_size
    except Exception:
        return 0


def clean_empty_directories(root_dir: str) -> int:
    """Remove all empty directories within a root directory.

    Parameters
    ----------
    root_dir : str
        Root directory to clean

    Returns
    -------
    int
        Number of directories removed
    """
    if not os.path.exists(root_dir):
        return 0

    removed_count = 0
    try:
        # Walk from bottom up to handle nested empty directories
        for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
            if not dirnames and not filenames and dirpath != root_dir:
                try:
                    os.rmdir(dirpath)
                    removed_count += 1
                    print(f"Removed empty directory: {dirpath}")
                except OSError:
                    continue
        return removed_count
    except Exception:
        return removed_count


def validate_path(path: str) -> bool:
    """Validate if a path is valid for the current operating system.

    Parameters
    ----------
    path : str
        Path to validate

    Returns
    -------
    bool
        True if path is valid
    """
    try:
        # Use pathlib for cross-platform validation
        Path(path)
        return True
    except (OSError, ValueError):
        return False


def create_directory_structure(base_path: str, structure: dict) -> bool:
    """Create a nested directory structure from a dictionary.

    Parameters
    ----------
    base_path : str
        Base directory path
    structure : dict
        Dictionary representing directory structure

    Example
    -------
    structure = {
        'data': {
            'raw': {},
            'processed': {}
        },
        'output': {
            'images': {},
            'files': {}
        }
    }

    Returns
    -------
    bool
        True if all directories were created successfully
    """
    try:

        def create_dirs(current_path: str, current_structure: dict):
            for name, subdirs in current_structure.items():
                new_path = os.path.join(current_path, name)
                dir_maker(new_path)
                if isinstance(subdirs, dict) and subdirs:
                    create_dirs(new_path, subdirs)

        create_dirs(base_path, structure)
        return True
    except Exception as e:
        print(f"Error creating directory structure: {str(e)}")
        return False
