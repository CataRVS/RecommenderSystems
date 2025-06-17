from abc import ABC, abstractmethod
from typing import Optional
from src.datamodule.data import AbstractData
import numpy as np


class Splitter(ABC):
    """
    Base class for train/test splitting strategies.

    This class defines the interface for splitting data into training and test sets.
    Subclasses should implement the `split` method.
    """

    @abstractmethod
    def split(self, data: AbstractData) -> AbstractData:
        """
        Split the data into training and test sets.

        Parameters:
            data (AbstractData): The data to be split.

        Returns:
            AbstractData: An AbstractData object containing the split data.
        """
        pass


class RandomSplitter(Splitter):
    """
    Splits the data randomly into training and test sets.
    This class implements a random splitting strategy where the data is randomly
    divided into training and test sets based on a specified test size.

    Attributes:
        test_size (float): Proportion of the dataset to include in the test split.
        seed (int, optional): Random seed for reproducibility.
    """
    def __init__(self, test_size: float = 0.2, seed: Optional[int] = None):
        """
        Initialize the splitter with a test size.

        Parameters:
            test_size (float): Proportion of the dataset to include in the test split.
            seed (int, optional): Random seed for reproducibility.
        """
        self.test_size = test_size
        self.seed = seed

    def split(self, data: AbstractData) -> AbstractData:
        train_data, test_data = data.get_train_test_data()
        combined_data = train_data + test_data

        n_test = int(len(combined_data) * self.test_size)

        # If n_test is zero, return empty test set
        if n_test == 0:
            return data.from_train_test_data(combined_data, [])

        # Set the random seed for reproducibility
        if self.seed is not None:
            np.random.seed(self.seed)

        # Shuffle the combined data and split based on test proportion
        np.random.shuffle(combined_data)
        train_result = combined_data[:-n_test]
        test_result = combined_data[-n_test:]

        return data.from_train_test_data(train_result, test_result)


class LeaveOneLastSplitter(Splitter):
    """
    Splits the data by leaving out the last interaction for each user.

    This class implements a leave-one-last strategy where the last interaction
    of each user is used as the test set, and all previous interactions are used
    for training.
    """

    def split(self, data: AbstractData) -> AbstractData:
        """
        Split the data by leaving out the last interaction for each user.
        
        Parameters:
            data (AbstractData): The data to be split.

        Returns:
            AbstractData: An AbstractData object containing the split data.
        """
        train_data, test_data = data.get_train_test_data()
        combined_data = train_data + test_data

        interactions_per_user = {}
        for user, item, rating, timestamp in combined_data:
            interactions_per_user.setdefault(user, []).append((item, rating, timestamp))

        train_result = []
        test_result = []

        for user, interactions in interactions_per_user.items():
            interactions.sort(key=lambda x: x[2])  # Sort by timestamp
            if len(interactions) > 1:
                for item, rating, timestamp in interactions[:-1]:
                    train_result.append((user, item, rating, timestamp))
                last_item, last_rating, last_timestamp = interactions[-1]
                test_result.append((user, last_item, last_rating, last_timestamp))
            else:
                # If only one interaction, put it in test set
                item, rating, timestamp = interactions[0]
                test_result.append((user, item, rating, timestamp))

        return data.from_train_test_data(train_result, test_result)


class TemporalUserSplitter(Splitter):
    """
    Splits the data based on user interactions over time.

    This class implements a temporal user splitting strategy where for each user,
    interactions are sorted by timestamp, and the latest `test_size` proportion
    is placed in the test set. Ensures user-wise temporal consistency.

    Attributes:
        test_size (float): Proportion of the dataset to include in the test split.
    """
    def __init__(self, test_size: float = 0.2):
        """
        Initialize the splitter with a test size.

        Parameters:
            test_size (float): Proportion of the dataset to include in the test split.
        """
        self.test_size = test_size

    def split(self, data: AbstractData) -> AbstractData:
        train_data, test_data = data.get_train_test_data()
        combined_data = train_data + test_data

        interactions_per_user = {}
        for user, item, rating, timestamp in combined_data:
            interactions_per_user.setdefault(user, []).append((item, rating, timestamp))

        train_result = []
        test_result = []

        for user, interactions in interactions_per_user.items():
            interactions.sort(key=lambda x: x[2])
            n_test = int(len(interactions) * self.test_size)
            if n_test == 0:
                continue

            for i, (item, rating, timestamp) in enumerate(interactions):
                if i < len(interactions) - n_test:
                    train_result.append((user, item, rating, timestamp))
                else:
                    test_result.append((user, item, rating, timestamp))

        return data.from_train_test_data(train_result, test_result)


class TemporalGlobalSplitter(Splitter):
    """
    Splits all interactions based on global timestamp.

    This class implements a temporal global splitting strategy where all interactions
    are sorted chronologically, the earliest `(1 - test_size)` proportion is used for
    training, and the rest for testing. Useful for evaluating models in time-aware scenarios.

    Attributes:
        test_size (float): Proportion of the dataset to include in the test split.
    """
    def __init__(self, test_size: float = 0.2):
        """
        Initialize the splitter with a test size.

        Parameters:
            test_size (float): Proportion of the dataset to include in the test split.
        """
        self.test_size = test_size

    def split(self, data: AbstractData) -> AbstractData:
        train_data, test_data = data.get_train_test_data()
        combined_data = train_data + test_data

        # Sort all interactions by global timestamp
        sorted_data = sorted(combined_data, key=lambda x: x[3])

        cutoff_index = int((1 - self.test_size) * len(sorted_data))
        train_result = sorted_data[:cutoff_index]
        test_result = sorted_data[cutoff_index:]

        return data.from_train_test_data(train_result, test_result)
