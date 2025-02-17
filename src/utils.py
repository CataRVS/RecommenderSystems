from abc import ABC, abstractmethod
from src.data import Data
from typing import List, Tuple


class Strategy(ABC):
    """
    Abstract class to define recommendation filtering strategies.
    """

    @abstractmethod
    def filter(self, user_id: int, data: Data) -> list:
        """
        Filter the candidate items to recommend.

        Parameters:
            user_id (int): User ID.
            data (Data): Data instance with the user-item interactions.

        Returns:
            list: List of filtered items.
        """
        pass


class ExcludeSeenStrategy(Strategy):
    """
    Strategy that excludes items already seen by the user.
    """

    def filter(self, user_id: int, data: Data) -> list:
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
        user_history = data.get_user_ratings(user_id, original_id=True)

        # TODO: Why does it return a generator?
        # We filter the candidates
        # FIXME: We are not abstracting the data structure here
        candidates = [item for item in range(data.get_total_items()) if item not in user_history["item"].values]
        return candidates


class Recommendation:
    """
    Represents a recommendation for a user.

    Attributes:
        user_id (int): Original user ID.
        recommended_items (list): List of recommended items.
    """

    def __init__(self, user_id: int, recommended_items: List[tuple]):
        """
        Create a new Recommendation instance.

        Parameters:
            user_id (int): Original user ID.
            recommended_items (list): List of recommended items. Each item is a tuple
                with the item ID and the recommendation score.
        """
        self.user_id = user_id
        self.recommended_items = recommended_items

    def __str__(self) -> str:
        """
        Get the string representation of the recommendation.

        Returns:
            str: String representation of the recommendation.
        """
        text = f"User {self.user_id} - Recommendations:\n"
        for item, score in self.recommended_items:
            text += f"+ {item} - {score}\n"

        return text

    def get_user_id(self) -> int:
        """
        Get the original user ID.

        Returns:
            int: Original user ID.
        """
        return self.user_id

    def get_recommendation(self) -> List[Tuple[str, float]]:
        """
        Get the list of recommended items.

        Returns:
            list: List of recommended items. Each item is a tuple with the item ID (str)
                and the recommendation score (float).
        """
        return self.recommended_items
