from abc import ABC, abstractmethod
from src.datamodule.data import Data
from src.utils.utils import Recommendation
from src.utils.strategies import Strategy
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
        self,
        user_id: int,
        strategy: Strategy,
        recommendation: Recommendation = Recommendation(),
        n: int = 10,
    ) -> Recommendation:
        """
        Recommend items to the user.

        Parameters:
            user_id (int): Original user ID.
            strategy (Strategy): Recommendation strategy.
            recommendation (Recommendation): Recommendation instance to add the
                recommendations to.
            n (int): Number of items to recommend.

        Returns:
            Recommendation: Recommendation instance with the top-n recommendations for
                the user.
        """
        pass


class PopularityRecommender(Recommender):
    """
    Recommends the most popular items (most consumed).

    Attributes:
        popularity_scores (dict): Popularity score (count) of each item.
    """

    def __init__(self, data: Data):
        """
        Create a new PopularityRecommender instance.

        Parameters:
            data (Data): Data instance with the user-item interactions.
        """
        # Call the parent constructor
        super().__init__(data)
        # Compute the popularity scores of the items with the training data
        self.popularity_scores = self._compute_popularity()

    def _compute_popularity(self) -> dict:
        """
        Compute the popularity of each item in the dataset.

        Returns:
            dict: Popularity score (count) of each item.
        """
        # Initialize the popularity scores
        popularity_scores = {}
        # Get the candidates to recommend
        candidates = self.data.get_items()

        # Compute the popularity scores for each candidate based on the number of
        # interactions
        for candidate in candidates:
            # Get the number of interactions for the item
            popularity_scores[candidate] = len(
                self.data.get_interaction_from_item(candidate)
            )

        return popularity_scores

    def recommend(
        self,
        user_id: int,
        strategy: Strategy,
        recommendation: Recommendation = Recommendation(),
        n: int = 10,
    ) -> Recommendation:
        """
        Recommend the most popular items to the user.

        Parameters:
            user_id (int): Original user ID.
            strategy (Strategy): Recommendation strategy.
            recommendation (Recommendation): Recommendation instance to add the
                recommendations to.
            n (int): Number of items to recommend.

        Returns:
            Recommendation: Recommendation instance with the top-n recommendations for
                the user.
        """
        # Filter the candidates based on the strategy
        filtered_candidates = strategy.filter(user_id)

        # Sort the filtered candidates by popularity score
        sorted_candidates = sorted(
            filtered_candidates,
            key=lambda item: self.popularity_scores[item],
            reverse=True,
        )

        # Take the top-n items
        top_candidates = sorted_candidates[:n]

        # Create the recommendations item list
        recommendations = [
            (item, self.popularity_scores[item]) for item in top_candidates
        ]

        # Add the recommendations to the Recommendation instance
        recommendation.add_recommendations(user_id, recommendations)
        return recommendation


class RandomRecommender(Recommender):
    """
    Recommends random items to users.
    """

    def recommend(
        self,
        user_id: int,
        strategy: Strategy,
        recommendation: Recommendation = Recommendation(),
        n: int = 10,
    ) -> Recommendation:
        """
        Recommend random items to the user.

        Parameters:
            user_id (int): Original user ID.
            strategy (Strategy): Recommendation strategy.
            recommendation (Recommendation): Recommendation instance to add the
                recommendations to.
            n (int): Number of items to recommend.

        Returns:
            Recommendation: Recommendation instance with the top-n recommendations for
                the user.
        """
        # Filter the candidates based on the strategy
        filtered_candidates = strategy.filter(user_id)

        # Take n random items from the filtered candidates
        recommended_items = random.sample(
            filtered_candidates, min(n, len(filtered_candidates))
        )

        # Create the recommendations item list
        recommendation_items = [
            (item, n - i) for i, item in enumerate(recommended_items)
        ]

        # Add the recommendations to the Recommendation instance
        recommendation.add_recommendations(user_id, recommendation_items)
        return recommendation
