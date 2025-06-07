from src.recommenders.basic_recommenders import Recommender
from src.utils.utils import Recommendation
from src.utils.similarities import Similarity, PearsonCorrelationUsers, PearsonCorrelationItems
from src.datamodule.data import Data
from src.utils.strategies import Strategy
from typing import List, Tuple
import numpy as np


class KNNUserBasedRecommender(Recommender):
    """
    KNN user-based collaborative filtering recommender system.

    Attributes:
        data (Data): Data instance with the user-item interactions.
        k (int): Number of similar users to consider for prediction.
        threshold (float): Rating threshold above which a neighbor “votes” for the item.
        similarity (Similarity): Similarity measure to use for finding similar users.
    """
    def __init__(
        self,
        data: Data,
        k: int = 10,
        similarity_measure: Similarity | None = None,
        threshold: float = 4.0
    ) -> None:
        """
        Create a new KNNUserBasedRecommender instance.

        Parameters:
            data (Data): Data instance with the user-item interactions.
            k (int): Number of similar users to consider for prediction.
            threshold (float): Rating threshold above which a neighbor “votes” for the item.
            similarity_measure (Similarity|None): Similarity measure to use for finding
                similar users. If None, the default PearsonCorrelationSimilarity is used.
        """
        super().__init__(data)
        self.k = k
        self.threshold = threshold
        # If there is no similarity measure, use the default one
        if similarity_measure is None:
            # user–user similarity
            self.similarity = PearsonCorrelationUsers(data)
        else:
            self.similarity = similarity_measure

    def recommend(
        self,
        user_id: int,
        strategy: Strategy,
        recommendation: Recommendation = Recommendation(),
        n: int = 10
    ) -> Recommendation:
        # For each candidate item i:
        #   1. Find the k most similar users to user_id who have rated i
        #   2. Let votes_v = 1 if r_{v,i} >= threshold else 0
        #   3. score(i) = sum_{v in N_k} (w_{u,v} * votes_v)

        # Filter the candidates based on the strategy
        candidates: List[int] = strategy.filter(user_id)

        # We initialize the list of recommendations
        predictions: List[Tuple[int, float]] = []

        # Convert external user ID to internal index
        u_idx = self.data.to_internal_user(user_id)

        for item in candidates:
            # Retrieve internal users and their ratings for this item
            users_idx, ratings = self.data.get_item_interactions_indices(item)

            # Get the precomputed similarities for user u_idx from the similarity class
            sims = self.similarity.sim_matrix[u_idx, users_idx]
            # only positive contributions
            sims = np.clip(sims, 0, None)

            # Select top-k neighbors by similarity magnitude
            top_k_indices = np.argsort(-np.abs(sims))[: self.k]  # Top-k indices
            top_sims = sims[top_k_indices]  # Top-k similarities
            top_rats = ratings[top_k_indices]  # Top-k ratings

            # Compute the votes based on the threshold
            # votes_v = 1 if r_{v,i} >= threshold else 0
            votes = (top_rats >= self.threshold).astype(float)

            # We compute the score as the weighted sum of votes
            score = float(np.dot(top_sims, votes))

            # Add the score to the predictions list
            predictions.append((item, score))

        # Sort the recommendations by predicted rating in descending order
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Take the top-n items from the sorted list (if there are less than n items, take all)
        if len(predictions) > n:
            top_n_recommendations = predictions[:n]
        else:
            top_n_recommendations = predictions

        # Add the recommendations to the Recommendation instance
        recommendation.add_recommendations(user_id, top_n_recommendations)
        return recommendation


class KNNItemBasedRecommender(Recommender):
    """
    KNN item-based collaborative filtering recommender system.

    Attributes:
        data (Data): Data instance with the user-item interactions.
        k (int): Number of similar items to consider for prediction.
        similarity (Similarity): Similarity measure to use for finding similar items.
    """
    def __init__(
        self, data: Data, k: int = 10, similarity_measure: Similarity | None = None
    ) -> None:
        """
        Create a new KNNItemBasedRecommender instance.

        Parameters:
            data (Data): Data instance with the user-item interactions.
            k (int): Number of similar items to consider for prediction.
            similarity_measure (Similarity | None): Similarity measure to use for finding
                similar items. If None, the default PearsonCorrelationSimilarity is used.
        """
        super().__init__(data)
        self.k = k
        # If there is no similarity measure, use the default one
        if similarity_measure is None:
            # item–item similarity
            self.similarity = PearsonCorrelationItems(data)
        else:
            self.similarity = similarity_measure

    def recommend(
        self,
        user_id: int,
        strategy: Strategy,
        recommendation: Recommendation = Recommendation(),
        n: int = 10
    ) -> Recommendation:
        # Filter the candidates based on the strategy
        candidates: List[int] = strategy.filter(user_id)
        predictions: List[Tuple[int, float]] = []

        # Obtain the active user's history and their average rating (to use as a fallback)
        user_history = self.data.get_interaction_from_user(user_id)  # {item: rating}
        # If the user has rated items, calculate the average rating
        user_avg = (sum(user_history.values()) / len(user_history)) if user_history else 0.0

        # For each candidate item, we calculate the predicted rating
        for item in candidates:
            # Get the items that the user has rated
            user_items_idx, ratings = self.data.get_user_interactions_indices(user_id)

            # Get the precomputed similarities for item from the similarity class
            item_idx = self.data.to_internal_item(item)
            item_sims = self.similarity.sim_matrix[item_idx, user_items_idx]

            # Select top-k neighbors by similarity magnitude
            top_k_indices = np.argsort(-np.abs(item_sims))[: self.k]  # Top-k indices
            top_sims = item_sims[top_k_indices]  # Top-k similarities
            top_rats = ratings[top_k_indices]  # Top-k ratings

            # Compute the predicted rating using the weighted average of the neighbors
            # r̂_ui = (Σ_v w_uv * r_vi) / (Σ_v |w_uv|)
            # where w_uv is the similarity between item u and item v, and r_vi is the rating
            # given by user u to item v
            numerator = np.dot(top_sims, top_rats)
            denominator = np.sum(np.abs(top_sims))
            # If there are no neighbors, use the user's average rating
            predicted_rating = (numerator / denominator) if denominator > 0 else user_avg

            # Add the predicted rating to the recommendations list
            predictions.append((item, predicted_rating))

        # Sort the recommendations by predicted rating in descending order
        predictions.sort(key=lambda x: x[1], reverse=True)
        # Take the top-n items from the sorted list (if there are less than n items, take all)
        if len(predictions) > n:
            top_n_recommendations = predictions[:n]
        else:
            top_n_recommendations = predictions

        # Add the recommendations to the Recommendation instance
        recommendation.add_recommendations(user_id, top_n_recommendations)
        return recommendation
