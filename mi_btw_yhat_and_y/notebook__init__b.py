# File: notebooks/notebook_init.py
import os
import sys
from pathlib import Path


def setup_project_environment() -> None:
    """Set up the jupyter notebook path with the root folder as the main dir."""
    # Get the project root directory
    project_root = Path(__file__).resolve().parent.parent
    # project_root = Path(__file__).resolve().parent
    print(project_root)
    print(Path(__file__).resolve().parent)

    # Add the project root and utils directory to the Python path
    paths_to_add = [str(project_root), str(project_root / "utils")]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

    # Change the working directory to the project root
    os.chdir(project_root)

    # Now you can import your project utilities
    # Note: This is for the new created directory only not for this dir uses
    from path_utils.path_handler import ProjectPaths

    # Set the project root
    ProjectPaths.set_project_root(project_root)

    print(f"Project root set to: {ProjectPaths.get_project_root()}")
    print("Project root and utils directory added to Python path.")
    print("Working directory changed to project root.")

    # Return the ProjectPaths class for convenience
    return ProjectPaths


# Run the setup
ProjectPaths = setup_project_environment()
