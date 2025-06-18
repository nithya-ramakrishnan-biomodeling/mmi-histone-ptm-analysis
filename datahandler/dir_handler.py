import os


def dir_maker(full_path: str) -> None:
    """Create new directory from the given path.

    Parameters
    ----------
    full_path : str
        Absolute path of the new directory.
    """

    # if os.path.isdir(full_path)==False:

    #     # splitting the path into dir and file
    #     dir_path, _ =  os.path.split(full_path)

    # else:
    #     dir_path = full_path

    os.makedirs(full_path, exist_ok=True)


def dir_path_generator(cur_dir, backstep_num: int) -> str:
    """Generate path for the main project

    Parameter:
    ----------
    backstep_num : int
    Provide integer value that represents
    how many steps are we need to take to reach
    the main directory.

    Returns
    -------
    str
        Main directory path
    """

    no_of_steps = [".."] * backstep_num
    main_dir = os.path.abspath(os.path.join(cur_dir, *no_of_steps))

    return main_dir
