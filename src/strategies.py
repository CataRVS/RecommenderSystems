from abc import ABC, abstractmethod
from src.data import Data
from typing import List


class Strategy(ABC):
    """
    Abstract class to define recommendation filtering strategies.
    """

    def __init__(self, data: Data):
        """
        Create a new Strategy instance.

        Parameters:
            data (Data): Data instance with the user-item interactions.
        """
        self.data = data

    @abstractmethod
    def filter(self, user_id: int) -> List[int]:
        """
        Filter the candidate items to recommend.

        Parameters:
            user_id (int): User ID.

        Returns:
            list: List of filtered items.
        """
        pass


class ExcludeSeenStrategy(Strategy):
    """
    Strategy that excludes items already seen by the user.
    """

    def __init__(self, data: Data):
        """
        Create a new ExcludeSeenStrategy instance.

        Parameters:
            data (Data): Data instance with the user-item interactions.
        """
        # Call the parent constructor
        super().__init__(data)

    def filter(self, user_id: int) -> List[int]:
        """
        Filter the items that the user has already seen.

        Parameters:
            user_id (int): Original user ID.
            candidates (list): Items to be filtered.
            user_history (dict): User interaction history.

        Returns:
            list: List of filtered items.
        """
        # We get the items already seen by the user
        user_history = set(self.data.get_interaction_from_user(user_id))

        # We filter the candidates
        candidates = [
            item for item in self.data.get_items() if item not in user_history
        ]
        return candidates


class NoFilteringStrategy(Strategy):
    """
    Strategy that does not apply any filtering.
    """

    def __init__(self, data: Data):
        """
        Create a new NoStrategy instance.

        Parameters:
            data (Data): Data instance with the user-item interactions.
        """
        # Call the parent constructor
        super().__init__(data)

    def filter(self, user_id: int) -> List[int]:
        """
        Return all the items as candidates.

        Parameters:
            user_id (int): Original user ID.
            candidates (list): Items to be filtered.
            user_history (dict): User interaction history.

        Returns:
            list: List of filtered items.
        """
        # Return all the items
        return self.data.get_items()
