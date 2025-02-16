import pandas as pd
import numpy as np
from typing import Tuple


class Data:
    """
    Data class for loading and preprocessing data.

    Attributes:
        total_users (int): Total number of users in the dataset.
        total_items (int): Total number of items in the dataset.
        uid_map (dict): Mapping of user ids to internal user ids.
        iid_map (dict): Mapping of item ids to internal item ids.
        uidx_map (dict): Mapping of internal user ids to user ids.
        iidx_map (dict): Mapping of internal item ids to item ids.
        train (pd.DataFrame): Training data.
        test (pd.DataFrame): Testing data.
    """

    def __init__(
            self,
            data_path: str,
            sep: str = "\t",
            test_size: float = 0.2,
            ignore_first_line: bool = False,
            col_names: list = ["user", "item", "rating", "timestamp"]
        ) -> None:
        """
        Constructor for Data class.

        Args:
            data_path (str): Path to the dataset.
            sep (str): Delimiter for the dataset.
            test_size (float): Proportion of data to split into test set.

        Raises:
            FileNotFoundError: If the dataset is not found.
        """
        # Initialize attributes
        self.total_users: int = 0
        self.total_items: int = 0
        self.uid_map: dict = {}
        self.iid_map: dict = {}
        self.uidx_map: dict = {}
        self.iidx_map: dict = {}

        # Load and preprocess data
        try:
            data: pd.DataFrame = pd.read_csv(
                data_path,
                sep=sep,
                header=None,
                names=col_names,
                skiprows=1 if ignore_first_line else 0
            )

        except FileNotFoundError:
            raise FileNotFoundError("File not found. Please change the path and try again.")

        except pd.errors.ParserError:
            raise pd.errors.ParserError("Error parsing the file. Please check the file format and try again.")

        self._preprocess(data, test_size)

    def _preprocess(self, data: pd.DataFrame, test_size: float) -> None:
        """
        Preprocess the data.

        Args:
            data (pd.DataFrame): Data to preprocess.
            test_size (float): Proportion of data to split into test set.
        """
        # Correctly name the columns
        data.columns = ["user", "item", "rating", "timestamp"]

        # Divide the data into training and testing sets
        data = data[["user", "item", "rating", "timestamp"]]

        # Create a mask for the test set
        test_mask = np.random.rand(len(data)) < test_size
        self.train = data[~test_mask]
        self.test = data[test_mask]

        # Update the total number of users and items
        self.total_users = self.train["user"].nunique()
        self.total_items = self.train["item"].nunique()

        # Create mappings for user and item ids
        self.uid_map = {uid: i for i, uid in enumerate(self.train["user"].unique())}
        self.iid_map = {iid: i for i, iid in enumerate(self.train["item"].unique())}

        # Create reverse mappings for user and item ids
        self.uidx_map = {i: uid for uid, i in self.uid_map.items()}
        self.iidx_map = {i: iid for iid, i in self.iid_map.items()}

        # Map the user and item ids to internal ids and save them
        self.train["user"] = self.train["user"].map(self.uid_map)
        self.train["item"] = self.train["item"].map(self.iid_map)

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the preprocessed data.

        Returns:
            tuple: Tuple containing training and testing data.
        """
        return self.train, self.test

    def get_mappings(self) -> Tuple[dict, dict]:
        """
        Get the mappings (from original ids to internal ids).

        Returns:
            tuple: Tuple containing user id mapping and item id mapping.
        """
        return self.uid_map, self.iid_map

    def get_reverse_mappings(self) -> Tuple[dict, dict]:
        """
        Get the reverse mappings (from internal ids to original ids).

        Returns:
            tuple: Tuple containing user id mapping and item id mapping.
        """
        return self.uidx_map, self.iidx_map

    def get_total_users(self) -> int:
        """
        Get the total number of users.

        Returns:
            int: Total number of users.
        """
        return self.total_users

    def get_total_items(self) -> int:
        """
        Get the total number of items.

        Returns:
            int: Total number of items.
        """
        return self.total_items

    def get_user_ratings(self, user_id: int, original_id: bool = False) -> pd.DataFrame:
        """
        Get the items rated by the user.

        Args:
            user_id (int): User id.
            original_id (bool): Whether the user id is the original or the internal id.

        Raises:
            KeyError: If the user id is not found.

        Returns:
            pd.DataFrame: Items rated by the user.
        """
        # Get the internal user id if the original id is provided
        if original_id:
            user_id = self.uid_map[user_id]

        # Return the items rated by the user
        return self.train[self.train["user"] == user_id]
