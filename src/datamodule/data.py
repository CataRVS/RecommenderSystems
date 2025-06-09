from abc import ABC, abstractmethod
from scipy.sparse import coo_matrix
from typing import Tuple
import numpy as np
import pandas as pd


class AbstractData(ABC):
    """
    Abstract class for loading and preprocessing data.

    Attributes:
        total_users (int): Total number of users in the dataset.
        total_items (int): Total number of items in the dataset.
        uid_map (dict): Mapping of user ids to internal user ids.
        iid_map (dict): Mapping of item ids to internal item ids.
        uidx_map (dict): Mapping of internal user ids to user ids.
        iidx_map (dict): Mapping of internal item ids to item ids.
        train: Training data.
        test: Testing data.
    """

    @abstractmethod
    def get_mappings(self) -> Tuple[dict, dict]:
        """
        Get the mappings (from original ids to internal ids).

        Returns:
            tuple: Tuple containing user id mapping and item id mapping.
        """
        pass

    @abstractmethod
    def get_reverse_mappings(self) -> Tuple[dict, dict]:
        """
        Get the reverse mappings (from internal ids to original ids).

        Returns:
            tuple: Tuple containing user id mapping and item id mapping.
        """
        pass

    @abstractmethod
    def get_total_users(self) -> int:
        """
        Get the total number of users.

        Returns:
            int: Total number of users.
        """
        pass

    @abstractmethod
    def get_total_items(self) -> int:
        """
        Get the total number of items.

        Returns:
            int: Total number of items.
        """
        pass

    @abstractmethod
    def get_users(self, test: bool = False) -> list:
        """
        Get the list of users.

        Parameters:
            test (bool): Whether to get the users from the test set. If False, get the
                users from the training set.

        Returns:
            list: List of users.
        """
        pass

    @abstractmethod
    def get_items(self, test: bool = False) -> list:
        """
        Get the list of items.

        Parameters:
            test (bool): Whether to get the items from the test set. If False, get the
                items from the training set.

        Returns:
            list: List of items.
        """
        pass

    @abstractmethod
    def get_interactions(self) -> tuple:
        """
        Get the user-item interactions as a tuple of numpy arrays.

        Returns:
            tuple: User indices, item indices, and ratings.
        """
        pass

    @abstractmethod
    def get_interaction_from_user(self, user_id: int, original_id: bool = True) -> dict:
        """
        Get the dictionary with the pairs of items and ratings for the user.

        Parameters:
            user_id (int): User id.
            original_id (bool): Whether the user id is the original or the internal id.

        Returns:
            dict: Dictionary with the pairs of items and ratings for the user.
        """
        pass

    @abstractmethod
    def get_interaction_from_item(self, item_id: int, original_id: bool = True) -> dict:
        """
        Get the dictionary with the pairs of users and ratings for the item.

        Parameters:
            item_id (int): Item id.
            original_id (bool): Whether the item id is the original or the internal id.

        Returns:
            dict: Dictionary with the pairs of users and ratings for the item.
        """
        pass

    @abstractmethod
    def to_internal_user(self, orig_user_id: int) -> int:
        """
        Convert an external user ID to its internal index.

        Parameters:
            orig_user_id (int): Original user ID.

        Returns:
            int: Internal user index or None if not found.
        """
        pass

    @abstractmethod
    def to_internal_item(self, orig_item_id: int) -> int:
        """
        Convert an external item ID to its internal index.

        Parameters:
            orig_item_id (int): Original item ID.

        Returns:
            int: Internal item index or None if not found.
        """
        pass

    @abstractmethod
    def get_item_interactions_indices(self, orig_item_id: int) -> tuple:
        """
        Retrieve the internal user indices and their ratings for a given item.

        Parameters:
            orig_item_id (int): Original item ID.

        Returns:
            tuple: Arrays of user indices and corresponding ratings.
        """

    @abstractmethod
    def get_user_interactions_indices(self, orig_user_id: int) -> tuple:
        """
        Retrieve the internal item indices and their ratings for a given user.

        Parameters:
            orig_user_id (int): Original user ID.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of item indices and corresponding ratings.
        """
        pass

    @abstractmethod
    def get_test_interactions(self, user_id: int) -> dict:
        """
        Return the test-set interactions for a given user.

        Parameters:
            user_id (int): External user ID.

        Returns:
            dict[int, float]: Mapping item_id → rating from the test split.
        """
        pass

    @staticmethod
    @abstractmethod
    def load_recs(
        path: str,
        sep: str = ",",
        ignore_first_line: bool = False,
    ) -> dict:
        """
        Load the recommendations from a CSV file.

        Parameters:
            path (str): Path to the CSV file.
            sep (str): Separator used in the CSV file.
            ignore_first_line (bool): Whether to ignore the first line of the CSV file.

        Returns:
            dict: Dictionary with user IDs as keys and lists of recommended items as values.
        """
        pass


class Data(AbstractData):
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
        data_path_train: str,
        data_path_test: str,
        sep: str = "\t",
        test_size: float = 0.2,
        ignore_first_line: bool = False,
        col_names: list = ["user", "item", "rating", "timestamp"],
    ) -> None:
        """
        Constructor for Data class.

        Parameters:
            data_path (str): Path to the dataset.
            sep (str): Delimiter for the dataset.
            test_size (float): Proportion of data to split into test set.

        Raises:
            FileNotFoundError: If the dataset is not found.
            pd.errors.ParserError: If there is an error parsing the dataset.
            ValueError: If the data is not in the expected format.
        """
        # Initialize attributes
        self._total_users: int = 0
        self._total_items: int = 0
        self._uid_map: dict = {}
        self._iid_map: dict = {}
        self._uidx_map: dict = {}
        self._iidx_map: dict = {}

        # Load and preprocess data
        try:
            if data_path_test == "none":
                data: pd.DataFrame = pd.read_csv(
                    data_path_train,
                    sep=sep,
                    header=None,
                    skiprows=1 if ignore_first_line else 0,
                )

                # Divide the data into training and testing sets
                train, test = self._divide_data(data, test_size)

            else:
                # The data is already divided into train and test sets
                # Load the training and testing data
                train: pd.DataFrame = pd.read_csv(
                    data_path_train,
                    sep=sep,
                    header=None,
                    skiprows=1 if ignore_first_line else 0,
                )
                test: pd.DataFrame = pd.read_csv(
                    data_path_test,
                    sep=sep,
                    header=None,
                    names=col_names,
                    skiprows=1 if ignore_first_line else 0,
                )

            # Preprocess the data
            self._train = self._preprocess(train, train=True)
            self._test = self._preprocess(test, train=False)

        except FileNotFoundError:
            raise FileNotFoundError(
                "File not found. Please change the path and try again."
            )

        except pd.errors.ParserError:
            raise pd.errors.ParserError(
                "Error parsing the file. Please check the file format and try again."
            )

        except ValueError:
            raise ValueError(
                "Data is not in the expected format. Please check the data and try again."
            )

    def _divide_data(
        self, data: pd.DataFrame, test_size: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide the data into training and testing sets.

        Parameters:
            data (pd.DataFrame): Data to divide.
            test_size (float): Proportion of data to split into test set.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and testing sets.
        """
        # Create a mask for the test set
        test_mask = np.random.rand(len(data)) < test_size
        train_data = data[~test_mask]
        test_data = data[test_mask]
        return train_data, test_data

    def _preprocess(self, data: pd.DataFrame, train: bool = True) -> pd.DataFrame:
        """
        Preprocess a data set.

        Parameters:
            data (pd.DataFrame): Data to preprocess.
            train (bool): Whether the data is training data or not.

        Returns:
            pd.DataFrame: Preprocessed data.

        Raises:
            ValueError: If the data is not in the expected format.
        """
        # Take only the first 4 columns of the data and rename them
        data = data.iloc[:, :4].copy()
        data.columns = ["user", "item", "rating", "timestamp"]

        if train:
            # Update the total number of users and items
            self._total_users = data["user"].nunique()
            self._total_items = data["item"].nunique()

            # Create mappings for user and item ids
            self._uid_map = {uid: i for i, uid in enumerate(data["user"].unique())}
            self._iid_map = {iid: i for i, iid in enumerate(data["item"].unique())}

            # Create reverse mappings for user and item ids
            self._uidx_map = {i: uid for uid, i in self._uid_map.items()}
            self._iidx_map = {i: iid for iid, i in self._iid_map.items()}

            # Map the user and item ids to internal ids and save them
            data.loc[:, "user"] = data["user"].map(self._uid_map)
            data.loc[:, "item"] = data["item"].map(self._iid_map)

        # Return the preprocessed data
        return data

    def _get_train_sparse_matrix(self):
        """
        Create a sparse matrix from the train data with users as rows and items as columns.

        Returns:
            scipy.sparse.csr_array: Sparse matrix.
        """
        # Create a sparse matrix from the data with users as rows and items as columns

        # Get the rows, columns and data for the sparse matrix
        rows = self._train["user"].values  # User ids
        cols = self._train["item"].values  # Item ids
        data = self._train["rating"].values  # Ratings

        # Create the sparse matrix
        sparse_matrix = coo_matrix(
            (data, (rows, cols)), shape=(self._total_users, self._total_items)
        )

        # Convert to CSR format for efficient arithmetic and matrix-vector operations
        return sparse_matrix.tocsr()

    def get_mappings(self) -> Tuple[dict, dict]:
        return self._uid_map, self._iid_map

    def get_reverse_mappings(self) -> Tuple[dict, dict]:
        return self._uidx_map, self._iidx_map

    def get_total_users(self) -> int:
        return self._total_users

    def get_total_items(self) -> int:
        return self._total_items

    def get_users(self, test: bool = False) -> list:
        if test:
            # Get the users from the test set
            return list(self._test["user"].unique())
        return list(self._uid_map.keys())

    def get_items(self, test: bool = False) -> list:
        if test:
            # Get the items from the test set
            return list(self._test["item"].unique())
        return list(self._iid_map.keys())

    def get_interactions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_coo = self._get_train_sparse_matrix().tocoo()

        # Convert rows and columns to numpy arrays
        return (train_coo.row, train_coo.col, train_coo.data)

    def get_interaction_from_user(self, user_id: int, original_id: bool = True,) -> dict:
        # Get the internal user id if the original id is provided
        if original_id:
            user_id = self._uid_map.get(user_id, None)
            if user_id is None:
                return {}

        # Get the items rated by the user
        user_ratings = self._train[self._train["user"] == user_id]

        # If we're not using the internal id, we need to map the item ids back to the original ids
        if original_id:
            user_ratings.loc[:, "item"] = user_ratings["item"].map(self._iidx_map)

        # Create a dictionary with the pairs of items and ratings
        user_ratings_dict = dict(zip(user_ratings["item"], user_ratings["rating"]))
        return user_ratings_dict

    def get_interaction_from_item(self, item_id: int, original_id: bool = True) -> dict:
        # Get the internal item id if the original id is provided
        if original_id:
            item_id = self._iid_map.get(item_id, None)
            if item_id is None:
                return {}

        # Get the users who rated the item
        item_ratings = self._train[self._train["item"] == item_id]

        # If we're not using the internal id, we need to map the user ids back to the original ids
        if original_id:
            item_ratings.loc[:, "user"] = item_ratings["user"].map(self._uidx_map)

        # Create a dictionary with the pairs of users and ratings
        item_ratings_dict = dict(zip(item_ratings["user"], item_ratings["rating"]))
        return item_ratings_dict

    def to_internal_user(self, orig_user_id: int) -> int:
        return self._uid_map.get(orig_user_id)

    def to_internal_item(self, orig_item_id: int) -> int:
        return self._iid_map.get(orig_item_id)

    def get_item_interactions_indices(self, orig_item_id: int) -> Tuple[np.ndarray, np.ndarray]:
        # Get the internal item id
        item_idx = self.to_internal_item(orig_item_id)
        # If the item id is not found, return empty arrays
        if item_idx is None:
            return np.array([], dtype=int), np.array([], dtype=float)
        # Get the user indices and ratings for the item
        col = self._get_train_sparse_matrix()[:, item_idx].tocoo()
        # Return the user indices and ratings
        return col.row, col.data

    def get_user_interactions_indices(self, orig_user_id: int) -> Tuple[np.ndarray, np.ndarray]:
        # Get the internal user id
        user_idx = self.to_internal_user(orig_user_id)
        # If the user id is not found, return empty arrays
        if user_idx is None:
            return np.array([], dtype=int), np.array([], dtype=float)
        # Get the item indices and ratings for the user
        row = self._get_train_sparse_matrix().getrow(user_idx).tocoo()
        # Return the item indices and ratings
        return row.col, row.data

    def get_test_interactions(self, user_id: int) -> dict:
        # Get the test set
        df = self._test
        # Select only rows for this external user
        user_df = df[df["user"] == user_id]
        # Build and return a dict {item: rating}
        return dict(zip(user_df["item"], user_df["rating"]))

    @staticmethod
    def load_recs(
        path: str,
        sep: str = ",",
        ignore_first_line: bool = False,
    ) -> dict:
        # Read the csv file and return the recommendations in a dictionary
        # returns {user_id: [item1, item2, …]}
        df = pd.read_csv(
            path,
            sep=sep,
            skiprows=1 if ignore_first_line else 0,
            header=None,
        )
        df.columns = ["user", "item", "score"]
        recs = {}
        for u, group in df.groupby("user"):
            recs[u] = group.sort_values("score", ascending=False)["item"].tolist()
        return recs
