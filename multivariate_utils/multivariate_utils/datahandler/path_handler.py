# File: project_utils/paths.py
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Union

# File: project_utils/paths.py


class ProjectPaths:
    "it is used to manage project directory paths"

    _PROJECT_ROOT: Optional[Path] = None

    @classmethod
    def set_project_root(cls, path: Union[str, Path]) -> None:
        cls._PROJECT_ROOT = Path(path)

    @classmethod
    def get_project_root(cls) -> Path:
        if cls._PROJECT_ROOT is None:
            cls._PROJECT_ROOT = Path.cwd()
        return cls._PROJECT_ROOT

    @classmethod
    def get_data_dir(cls) -> Path:
        return cls.get_project_root() / "data"

    @classmethod
    def get_raw_data_dir(cls) -> Path:
        return cls.get_data_dir() / "raw"

    @classmethod
    def get_cleaned_data_dir(cls) -> Path:
        return cls.get_data_dir() / "processed"

    @classmethod
    def get_notebooks_dir(cls) -> Path:
        return cls.get_project_root() / "notebooks"

    @classmethod
    def get_notebook_data_dir(cls) -> Path:
        return cls.get_notebooks_dir() / "data"

    @classmethod
    def get_notebook_model_dir(cls) -> Path:
        return cls.get_notebooks_dir() / "model"

    @classmethod
    def get_notebook_visualization_dir(cls) -> Path:
        return cls.get_notebooks_dir() / "visualization"

    @classmethod
    def get_output_dir(cls) -> Path:
        return cls.get_project_root() / "output"

    @classmethod
    def get_output_files_dir(cls) -> Path:
        return cls.get_output_dir() / "files"

    @classmethod
    def get_output_images_dir(cls) -> Path:
        return cls.get_output_dir() / "images"

    @classmethod
    def get_tests_dir(cls) -> Path:
        return cls.get_project_root() / "tests"

    @classmethod
    def get_test_notebook_dir(cls) -> Path:
        return cls.get_tests_dir() / "notebook"

    @classmethod
    def get_utils_dir(cls) -> Path:
        return cls.get_project_root() / "utils"


def ensure_dir_exists(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_file_path(filename: str, directory_func: Callable[[], Path]) -> Path:
    return directory_func() / filename


def save_data(data: Any, filename: str, directory_func: Callable[[], Path]) -> None:
    file_path = get_file_path(filename, directory_func)
    ensure_dir_exists(file_path.parent)
    with file_path.open("w", encoding="utf-8") as f:
        f.write(str(data))


def load_data(filename: str, directory_func: Callable[[], Path]) -> Any:
    file_path = get_file_path(filename, directory_func)
    with file_path.open("r", encoding="utf-8") as f:
        return f.read()


# Add the project root to the Python path
def add_project_root_to_path() -> None:
    project_root = ProjectPaths.get_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.chdir(project_root)  # Change the working directory to the project root
