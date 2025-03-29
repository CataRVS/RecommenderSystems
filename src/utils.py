import numpy as np
import random
from typing import List, Tuple


def set_seed(seed: int):
    """
    Set the seed for reproducibility.

    Parameters:
        seed (int): Seed value.
    """
    # TEST 1: Test if this is the correct way to set the seed
    random.seed(seed)
    np.random.seed(seed)


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
        return "\n".join(
            f"{self.user_id},{item_id},{score}"
            for item_id, score in self.recommended_items
        ) + "\n"


    def get_user_id(self) -> int:
        """
        Get the original user ID.

        Returns:
            int: Original user ID.
        """
        return self.user_id

    def get_recommendation(self) -> List[Tuple[int, float]]:
        """
        Get the list of recommended items.

        Returns:
            list: List of recommended items. Each item is a tuple with the item ID (str)
                and the recommendation score (float).
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
