from abc import ABC, abstractmethod
from collections import Counter
from src.data import Data
from src.utils import Strategy, Recommendation
from typing import Dict, List, Tuple
import random


class Recommender(ABC):
    """
    Abstract base class for recommenders.

    Attributes:
        data (Data): Data instance with the user-item interactions.
    """

    def __init__(self, data: Data):
        """
        Create a new Recommender instance.

        Parameters:
            data (Data): Data instance with the user-item interactions.
        """
        self.data = data

    @abstractmethod
    def recommend(
        self, user_id: int, strategy: Strategy, k: int = 10
    ) -> Recommendation:
        """
        Recommend items to the user.

        Parameters:
            user_id (str): Original user ID.
            strategy (Strategy): Recommendation strategy.
            k (int): Number of items to recommend.

        Returns:
            Recommendation: Recommendation instance with the top-k recommendations for
                the user.
        """
        pass


class PopularityRecommender(Recommender):
    """
    Recommends the most popular items (most consumed).

    Attributes:
        popularity_scores (Counter): Counter with the popularity score of each item.
    """

    def __init__(self, data: Data):
        """
        Create a new PopularityRecommender instance.

        Parameters:
            data (Data): Data instance with the user-item interactions.
        """
        # Call the parent constructor
        super().__init__(data)
        # Get the candidates to recommend
        train, _ = data.get_data()
        # Compute the popularity scores of the items with the training data
        # FIXME: We are not abstracting the data structure here
        self.popularity_scores = self._compute_popularity(train["item"])

    def _compute_popularity(self, candidates):
        """
        Compute the popularity of each item in the dataset.

        Parameters:
            candidates (list): List of candidate items.

        Returns:
            Counter: Counter with the popularity score of each item.
        """
        # Count the number of interactions for each item
        return Counter(candidates)

    def recommend(
        self, user_id: int, strategy: Strategy, k: int = 10
    ) -> Recommendation:
        """
        Recommend the most popular items to the user.

        Parameters:
            user_id (str): Original user ID.
            strategy (Strategy): Recommendation strategy.
            k (int): Number of items to recommend.

        Returns:
            Recommendation: Recommendation instance with the top-k recommendations for
                the user.
        """
        # Filter the candidates based on the strategy
        filtered_candidates = strategy.filter(user_id, self.data)
        # Sort the filtered candidates by popularity score
        sorted_candidates = sorted(
            filtered_candidates,
            key=lambda item: self.popularity_scores[item],
            reverse=True,
        )
        # Take the top-k items
        top_candidates = sorted_candidates[:k]
        # Create the recommendations item list
        recommendations = [
            (item, self.popularity_scores[item]) for item in top_candidates
        ]
        # Return the recommendations in a Recommendation instance
        return Recommendation(user_id, recommendations)


class RandomRecommender(Recommender):
    """
    Recommends random items to users.
    """

    def recommend(
        self, user_id: int, strategy: Strategy, k: int = 10
    ) -> Recommendation:
        """
        Recommend random items to the user.

        Parameters:
            user_id (str): Original user ID.
            strategy (Strategy): Recommendation strategy.
            k (int): Number of items to recommend.

        Returns:
            Recommendation: Recommendation instance with the top-k recommendations for
                the user.
        """
        # Filter the candidates based on the strategy
        filtered_candidates = strategy.filter(user_id, self.data)
        # Take k random items from the filtered candidates
        recommended_items = random.sample(
            filtered_candidates, min(k, len(filtered_candidates))
        )
        # Create the recommendations item list
        recommendation_items = [
            (item, k - i) for i, item in enumerate(recommended_items)
        ]
        # Return the recommendations in a Recommendation instance
        return Recommendation(user_id, recommendation_items)
