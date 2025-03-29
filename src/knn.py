from src.recommenders import Recommender
from src.utils import Recommendation
from src.similarities import PearsonCorrelationSimilarity
from src.data import Data
from src.strategies import Strategy
from typing import List, Tuple


class UserBasedRatingPredictionRecommender(Recommender):
    """
    User-based collaborative filtering recommender system.

    This recommender predicts the rating of an item for a user based on the ratings
    given by similar users. It uses a similarity measure to find the most similar
    users and then computes a weighted average of their ratings for the item.

    Attributes:
        data (Data): Data instance with the user-item interactions.
        k (int): Number of similar users to consider for prediction.
        similarity (Similarity): Similarity measure to use for finding similar users.
    """
    def __init__(self, data: Data, k: int = 10, similarity_measure=None) -> None:
        """
        Create a new UserBasedRatingPredictionRecommender instance.

        Parameters:
            data (Data): Data instance with the user-item interactions.
            k (int): Number of similar users to consider for prediction.
            similarity_measure (Similarity): Similarity measure to use for finding
                similar users. If None, the default PearsonCorrelationSimilarity is used.
        """
        super().__init__(data)
        self.k = k
        # If there is no similarity measure, use the default one
        if similarity_measure is None:
            self.similarity = PearsonCorrelationSimilarity(data)
        else:
            self.similarity = similarity_measure

    def recommend(self, user_id: int, strategy: Strategy, n: int = 10) -> Recommendation:
        """
        Recommend items to the user using user-based collaborative filtering.

        Parameters:
            user_id (int): Original user ID.
            strategy (Strategy): Recommendation strategy.
            n (int): Number of items to recommend.

        Returns:
            Recommendation: Recommendation instance with the top-n recommendations for
                the user.
        """
        # Filter the candidates based on the strategy
        candidates: List[int] = strategy.filter(user_id)

        # We initialize the list of recommendations
        recommendations: List[Tuple[int, float]] = []

        # We get the active user's history and their average rating (to use as a fallback)
        user_ratings = self.data.get_interaction_from_user(user_id)
        # If the user has rated items, calculate the average rating
        if user_ratings:
            user_avg = sum(user_ratings.values()) / len(user_ratings)
        # If the user has not rated any items, set the average rating to 0
        else:
            user_avg = 0.0

        # For each candidate item, we calculate the predicted rating
        for item in candidates:
            # Obtain all the users that have rated this item
            item_ratings = self.data.get_interaction_from_item(item)
            neighbors = []
            for v, r_v in item_ratings.items():
                # If the user is the same as the active user, skip it
                if v == user_id:
                    continue
                # Calculate the similarity between the active user and the neighbor
                sim = self.similarity.compute_user_similarity(user_id, v)
                # We only consider neighbors with a non-zero similarity
                if sim != 0:
                    neighbors.append((v, sim, r_v))
            # Order the neighbors by similarity in descending order
            neighbors.sort(key=lambda x: abs(x[1]), reverse=True)
            # Select the top-k neighbors
            top_neighbors = neighbors[:self.k]

            # Calculate the predicted rating using the weighted average of the neighbors
            # r̂_ui = (Σ_v w_uv * r_vi) / (Σ_v |w_uv|)
            if top_neighbors:
                numerator = sum(sim * r for (_, sim, r) in top_neighbors)
                denominator = sum(abs(sim) for (_, sim, _) in top_neighbors)
                if denominator > 0:
                    predicted_rating = numerator / denominator
                else:
                    predicted_rating = user_avg
            else:
                # If there are no neighbors, use the user's average rating
                predicted_rating = user_avg

            # Add the predicted rating to the recommendations list
            recommendations.append((item, predicted_rating))

        # Sort the recommendations by predicted rating in descending order
        recommendations.sort(key=lambda x: x[1], reverse=True)

        # Take the top-n items
        top_recommendations = recommendations[:n]

        # DUDA 5: A veces el item tiene una puntuacion negativa, y no se si eso es correcto
        # Deberia clippear la puntuacion a 0? Da igual si la puntuacion es negativa?

        # Return the recommendations
        return Recommendation(user_id, top_recommendations)


class ItemBasedRecommendationRecommender(Recommender):
    """
    Item-based collaborative filtering recommender system."
    This recommender predicts the rating of an item for a user based on the ratings
    given by similar items. It uses a similarity measure to find the most similar
    items and then computes a weighted average of their ratings for the item.

    Attributes:
        data (Data): Data instance with the user-item interactions.
        k (int): Number of similar items to consider for prediction.
        similarity (Similarity): Similarity measure to use for finding similar items.
    """
    def __init__(self, data: Data, k: int = 10, similarity_measure=None) -> None:
        """
        Create a new ItemBasedRecommendationRecommender instance.

        Parameters:
            data (Data): Data instance with the user-item interactions.
            k (int): Number of similar items to consider for prediction.
            similarity_measure (Similarity): Similarity measure to use for finding
                similar items. If None, the default PearsonCorrelationSimilarity is used.
        """
        super().__init__(data)
        self.k = k
        # If there is no similarity measure, use the default one
        if similarity_measure is None:
            self.similarity = PearsonCorrelationSimilarity(data)
        else:
            self.similarity = similarity_measure

    def recommend(self, user_id: int, strategy: Strategy, n: int = 10) -> Recommendation:
        """
        Recommend items to the user using item-based collaborative filtering.

        Parameters:
            user_id (int): Original user ID.
            strategy (Strategy): Recommendation strategy.
            n (int): Number of items to recommend.

        Returns:
            Recommendation: Recommendation instance with the top-n recommendations for
                the user.
        """
        # Filter the candidates based on the strategy
        candidates: List[int] = strategy.filter(user_id)
        recommendations: List[Tuple[int, float]] = []

        # Obtain the active user's history and their average rating (to use as a fallback)
        user_history = self.data.get_interaction_from_user(user_id)  # {item: rating}
        if user_history:
            user_avg = sum(user_history.values()) / len(user_history)
        else:
            user_avg = 0.0

        # For each candidate item, we calculate the predicted rating
        for item in candidates:
            # Obtain all the users that have rated this item
            neighbors = []
            # Get the ratings of the item from all users
            for j, r_j in user_history.items():
                sim = self.similarity.compute_item_similarity(item, j)
                if sim != 0:
                    neighbors.append((j, sim, r_j))
            # Order the neighbors by similarity in descending order and select the top-k
            neighbors.sort(key=lambda x: abs(x[1]), reverse=True)
            top_neighbors = neighbors[:self.k]

            # Calculate the predicted rating using the weighted average of the neighbors
            if top_neighbors:
                numerator = sum(sim * r for (_, sim, r) in top_neighbors)
                denominator = sum(abs(sim) for (_, sim, _) in top_neighbors)
                if denominator > 0:
                    predicted_rating = numerator / denominator
                else:
                    predicted_rating = user_avg
            else:
                # If there are no neighbors, use the user's average rating
                predicted_rating = user_avg

            # Add the predicted rating to the recommendations list
            recommendations.append((item, predicted_rating))

        # Sort the recommendations by predicted rating in descending order
        recommendations.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = recommendations[:n]
        return Recommendation(user_id, top_recommendations)
