import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd

from utils.Logger import Logger


class FileSystemUtils:
    """
    A utility class for working with the file system. It provides methods to interact with files and directories,
    such as finding files with specific extensions, checking file creation times, moving files, and reading/writing
    data from/to text, CSV, and JSON files.

    Attributes:
    ----------
    home : str
        The home directory of the user.

    root : str
        The root directory used by the utility, defaults to home if not specified.

    logger : Logger
        A logger instance for logging operations.
    """

    def __init__(self, root: Union[str, Path, bytes] = None):
        """
        Initializes the file system utility with a specified root directory.

        Parameters:
        ----------
        root : Union[str, Path, bytes], optional (default=None)
            The root directory to use. If not provided, the user's home directory is used.
        """
        self.home = str(Path.home())
        self.root = root if root else self.home
        self.logger: Logger = Logger()

    def set_full_path(self, path: Union[str, Path, bytes]) -> Union[str, Path, bytes]:
        """
        Resolves the full path, ensuring it starts with the root directory.

        Parameters:
        ----------
        path : Union[str, Path, bytes]
            The relative or absolute path to resolve.

        Returns:
        -------
        Union[str, Path, bytes]
            The full path to the file or directory.
        """
        if str(path).startswith((self.home, self.root)):
            return path
        else:
            return os.path.join(self.root, path)

    def find_files_in_directory(self, path: Union[str, Path, bytes]) -> List[Union[str, Path, bytes]]:
        """
        Finds all files in a given directory.

        Parameters:
        ----------
        path : Union[str, Path, bytes]
            The directory path to search in.

        Returns:
        -------
        List[Union[str, Path, bytes]]
            A list of file paths in the directory.
        """
        full_path = self.set_full_path(path)
        try:
            if os.path.exists(full_path):
                return [file for file in Path(full_path).iterdir() if file.is_file()]
            else:
                raise FileNotFoundError(f"Path {full_path} does not exist.")
        except Exception as e:
            self.logger.info(f"Cannot read files from directory {full_path}. Error: {e}")
            return []

    def find_files_with_extension(self, path: Union[str, Path, bytes], extension: str) -> List[Union[str, Path, bytes]]:
        """
        Finds all files in a directory with a specific extension.

        Parameters:
        ----------
        path : Union[str, Path, bytes]
            The directory path to search in.

        extension : str
            The file extension to filter by (e.g., ".txt").

        Returns:
        -------
        List[Union[str, Path, bytes]]
            A list of file paths with the specified extension.
        """
        files = self.find_files_in_directory(path)
        return [file for file in files if file.suffix == extension] if files else []

    def is_file_newer_than(self, path: Union[str, Path, bytes], date: datetime) -> bool:
        """
        Checks if a file is newer than a specified date.

        Parameters:
        ----------
        path : Union[str, Path, bytes]
            The path to the file.

        date : datetime.datetime
            The date to compare the file's creation time against.

        Returns:
        -------
        bool
            True if the file is newer than the specified date, False otherwise.
        """
        full_path = self.set_full_path(path)
        creation_time = datetime.fromtimestamp(os.path.getctime(full_path)).date()
        return creation_time > date

    def find_files_newer_than(self, path: Union[str, Path, bytes], date: datetime) -> List[
        Union[str, Path, bytes]]:
        """
        Finds all files in a directory that are newer than a specified date.

        Parameters:
        ----------
        path : Union[str, Path, bytes]
            The directory path to search in.

        date : datetime.datetime
            The date to compare the file's creation time against.

        Returns:
        -------
        List[Union[str, Path, bytes]]
            A list of file paths that are newer than the specified date.
        """
        files = self.find_files_in_directory(path)
        return [file for file in files if self.is_file_newer_than(file, date)] if files else []

    def get_file_size(self, path: Union[str, Path, bytes]) -> float:
        """
        Gets the size of a file in megabytes.

        Parameters:
        ----------
        path : Union[str, Path, bytes]
            The path to the file.

        Returns:
        -------
        float
            The size of the file in MB.
        """
        full_path = self.set_full_path(path)
        bytes_size = os.path.getsize(full_path)
        return bytes_size / (1024 * 1024)

    def get_folders_in_directory(self, path: Union[str, Path, bytes]) -> List[Union[str, Path, bytes]]:
        """
        Finds all folders in a given directory.

        Parameters:
        ----------
        path : Union[str, Path, bytes]
            The directory path to search in.

        Returns:
        -------
        List[Union[str, Path, bytes]]
            A list of folder paths in the directory.
        """
        full_path = self.set_full_path(path)
        return [os.path.join(full_path, d) for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))]

    def move_files(self, src_path: Union[str, Path, bytes], dst_path: Union[str, Path, bytes]) -> None:
        """
        Moves a file from the source path to the destination path.

        Parameters:
        ----------
        src_path : Union[str, Path, bytes]
            The source file path.

        dst_path : Union[str, Path, bytes]
            The destination file path.

        Raises:
        ------
        FileNotFoundError:
            If the source or destination path does not exist.
        """
        full_src_path = self.set_full_path(src_path)
        full_dst_path = self.set_full_path(dst_path)
        dst_parent_path = os.path.dirname(full_dst_path)
        if os.path.exists(full_src_path) and os.path.exists(dst_parent_path):
            shutil.move(full_src_path, full_dst_path)
        else:
            raise FileNotFoundError(
                f"Source path {full_src_path} or destination path {dst_parent_path} does not exist.")

    def save_string_as_txt_file(self, text: str, path: Union[str, Path, bytes]) -> None:
        """
        Saves a string as a text file at the specified path.

        Parameters:
        ----------
        text : str
            The text content to save in the file.

        path : Union[str, Path, bytes]
            The path to save the text file.

        Raises:
        ------
        FileNotFoundError:
            If the parent directory of the path does not exist.
        """
        full_path = self.set_full_path(path)
        parent_dir = os.path.dirname(full_path)
        if os.path.exists(parent_dir):
            with open(full_path, 'w', encoding='utf-8') as file:
                file.write(text)
        else:
            raise FileNotFoundError(f"Path {parent_dir} does not exist.")

    def add_column_to_existing_csv(self, path: Union[str, Path, bytes], column: Union[pd.Series, np.array, list],
                                   column_name: str) -> None:
        """
        Adds a new column to an existing CSV file.

        Parameters:
        ----------
        path : Union[str, Path, bytes]
            The path to the CSV file.

        column : Union[pd.Series, np.array, list]
            The new column to add to the CSV file.

        column_name : str
            The name of the new column.

        Raises:
        ------
        ValueError:
            If the DataFrame and the new column do not have the same number of rows.
        FileNotFoundError:
            If the CSV file does not exist.
        """
        full_path = self.set_full_path(path)
        if os.path.exists(full_path):
            df_temp = pd.read_csv(full_path, sep=';')
            if len(column) == len(df_temp):
                df_temp[column_name] = column
                df_temp.to_csv(full_path, index=False)
            else:
                raise ValueError("DataFrame and new column must be the same size.")
        else:
            raise FileNotFoundError(f"Path {full_path} does not exist.")

    def add_dict_to_existing_json(self, path: Union[str, Path, bytes], new_data: dict) -> None:
        """
        Adds data to an existing JSON file.

        Parameters:
        ----------
        path : Union[str, Path, bytes]
            The path to the JSON file.

        new_data : dict
            A dictionary containing the new data to add to the JSON file.

        Raises:
        ------
        FileNotFoundError:
            If the JSON file does not exist.
        JSONDecodeError:
            If the JSON file is not valid.
        """
        full_path = self.set_full_path(path)
        try:
            with open(full_path, "r") as json_file:
                existing_data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {}

        for key, values in new_data.items():
            if key in existing_data:
                existing_data[key].extend(values)
            else:
                existing_data[key] = values

        with open(full_path, "w") as json_file:
            json.dump(existing_data, json_file, indent=4)