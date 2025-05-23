import numpy as np
import random
from typing import List, Tuple


def set_seed(seed: int):
    """
    Set the seed for reproducibility.

    Parameters:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)


class Recommendation:
    """
    Represents a recommendation for a user.

    Attributes:
        recommended_items (list): List of recommended items.
    """

    def __init__(self):
        """
        Create a new Recommendation instance.
        """
        self.recommended_items = list()

    def __str__(self) -> str:
        """
        Get the string representation of the recommendation.

        Returns:
            str: String representation of the recommendation.
        """
        return "\n".join(
            f"{user_id},{item_id},{score}"
            for user_id, item_id, score in self.recommended_items
        ) + "\n"

    def add_recommendations(self, user_id: int, recommendations: List[Tuple[int, float]]):
        """
        Add recommended items to the recommendation list.

        Parameters:
            user_id (int): Original user ID.
            recommendations (list): List of recommended items. Each item is a tuple
                with the item ID and the recommendation score.
        """
        for item_id, score in recommendations:
            self.recommended_items.append((user_id, item_id, score))

    def get_recommendation(self) -> List[Tuple[int, int, float]]:
        """
        Get the list of recommended items.

        Returns:
            list: List of recommended items. Each item is a tuple with the user ID (int),
                the item ID (str) and the recommendation score (float).
        """
        return self.recommended_items

    def save(self, path: str, mode: str = "w"):
        """
        Save the recommendation to a file.

        Parameters:
            path (str): Path to save the recommendation.
            mode (str): File opening mode, "w" to write and "a" to append.

        Raises:
            ValueError: If the mode is invalid.
        """
        if mode not in ["w", "a"]:
            raise ValueError("Invalid mode. Use 'w' to write and 'a' to append.")

        with open(path, mode) as f:
            f.write(str(self))
