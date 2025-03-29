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
    def get_data(self) -> tuple:
        """
        Returns the training and testing data.

        Returns:
            tuple: Tuple containing training and testing data.
        """
        pass

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

    # @abstractmethod
    # def get_user_ratings(self, user_id: int, original_id: bool = True):
    #     """
    #     Get the items rated by the user.

    #     Args:
    #         user_id (int): User id.
    #         original_id (bool): Whether the user id is the original or the internal id.

    #     Raises:
    #         KeyError: If the user id is not found.

    #     Returns:
    #         Items rated by the user.
    #     """
    #     pass

    @abstractmethod
    def get_users(self, test: bool=False) -> list:
        """
        Get the list of users.

        Args:
            test (bool): Whether to get the users from the test set. If False, get the
                users from the training set.

        Returns:
            list: List of users.
        """
        pass

    @abstractmethod
    def get_items(self, test: bool=False) -> list:
        """
        Get the list of items.

        Args:
            test (bool): Whether to get the items from the test set. If False, get the
                items from the training set.

        Returns:
            list: List of items.
        """
        pass

    @abstractmethod
    def get_interaction_from_user(self, user_id: int, original_id: bool = True) -> dict:
        """
        Get the dictionary with the pairs of items and ratings for the user.

        Args:
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

        Args:
            item_id (int): Item id.
            original_id (bool): Whether the item id is the original or the internal id.

        Returns:
            dict: Dictionary with the pairs of users and ratings for the item.
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
        data_path: str,
        sep: str = "\t",
        test_size: float = 0.2,
        ignore_first_line: bool = False,
        col_names: list = ["user", "item", "rating", "timestamp"],
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
        self._total_users: int = 0
        self._total_items: int = 0
        self._uid_map: dict = {}
        self._iid_map: dict = {}
        self._uidx_map: dict = {}
        self._iidx_map: dict = {}

        # Load and preprocess data
        try:
            data: pd.DataFrame = pd.read_csv(
                data_path,
                sep=sep,
                header=None,
                names=col_names,
                skiprows=1 if ignore_first_line else 0,
            )

        except FileNotFoundError:
            raise FileNotFoundError(
                "File not found. Please change the path and try again."
            )

        except pd.errors.ParserError:
            raise pd.errors.ParserError(
                "Error parsing the file. Please check the file format and try again."
            )

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
        self._train = data[~test_mask]
        self._test = data[test_mask]

        # Update the total number of users and items
        self._total_users = self._train["user"].nunique()
        self._total_items = self._train["item"].nunique()

        # Create mappings for user and item ids
        self._uid_map = {uid: i for i, uid in enumerate(self._train["user"].unique())}
        self._iid_map = {iid: i for i, iid in enumerate(self._train["item"].unique())}

        # Create reverse mappings for user and item ids
        self._uidx_map = {i: uid for uid, i in self._uid_map.items()}
        self._iidx_map = {i: iid for iid, i in self._iid_map.items()}

        # Map the user and item ids to internal ids and save them
        self._train["user"] = self._train["user"].map(self._uid_map)
        self._train["item"] = self._train["item"].map(self._iid_map)

    def _get_train_sparse_matrix(self):
        """
        Create a sparse matrix from the train data with users as rows and items as columns.
        XXX 3: Ver si se necesita con test

        Returns:
            scipy.sparse.csr_matrix: Sparse matrix.
        """
        # TEST 4: Ver si funciona esta funcion (y si se puede usar comodamente)
        # Create a sparse matrix from the data with users as rows and items as columns
        # Funcion para transformar en numpy --> sparse filas usuarios columnas items

        # Get the rows, columns and data for the sparse matrix
        rows = self._train["user"].values  # User ids
        cols = self._train["item"].values  # Item ids
        data = self._train["rating"].values  # Ratings

        # Create the sparse matrix
        sparse_matrix = coo_matrix((data, (rows, cols)), shape=(self._total_users, self._total_items))

        # Convert to CSR format for efficient arithmetic and matrix-vector operations
        return sparse_matrix.tocsr()


    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the preprocessed data.
        XXX 1: Ver si se necesita esta funcion (no abstrae la estructura de datos)

        Returns:
            tuple: Tuple containing training and testing data.
        """
        return self._train, self._test

    def get_mappings(self) -> Tuple[dict, dict]:
        return self._uid_map, self._iid_map

    def get_reverse_mappings(self) -> Tuple[dict, dict]:
        return self._uidx_map, self._iidx_map

    def get_total_users(self) -> int:
        return self._total_users

    def get_total_items(self) -> int:
        return self._total_items

    # def get_user_ratings(self, user_id: int, original_id: bool = True) -> pd.DataFrame:
    #     """
    #     Get the items rated by the user.
    #     XXX 2: Ver si se necesita esta funcion (no abstrae la estructura de datos)

    #     Args:
    #         user_id (int): User id.
    #         original_id (bool): Whether the user id is the original or the internal id.

    #     Raises:
    #         KeyError: If the user id is not found.

    #     Returns:
    #         pd.DataFrame: Items rated by the user.
    #     """
    #     # Es necesaria o vale con get_interaction_from_user?
    #     # Get the internal user id if the original id is provided
    #     if original_id:
    #         user_id = self._uid_map[user_id]

    #     # Return the items rated by the user
    #     return self._train[self._train["user"] == user_id]

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
            user_ratings["item"] = user_ratings["item"].map(self._iidx_map)

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
            item_ratings["user"] = item_ratings["user"].map(self._uidx_map)

        # Create a dictionary with the pairs of users and ratings
        item_ratings_dict = dict(zip(item_ratings["user"], item_ratings["rating"]))
        return item_ratings_dict
