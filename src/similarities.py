from abc import ABC, abstractmethod
from src.data import Data
import numpy as np
# DUDA 1: Donde incluir las similarities
# No se si poner esto aqui o en la clase de data para poder usar dataframes, numpy, etc
# O incluirlas en las dos y dejar esta por si acaso (pero en general usar la de data)

class Similarity(ABC):
    """
    Abstract base class for similarity measures.
    """
    def __init__(self, data: Data):
        """
        Create a new Similarity instance.

        Parameters:
            data (Data): Data instance with the user-item interactions.
        """
        self.data = data

    @abstractmethod
    def compute_user_similarity(self, userU: int, userV: int) -> float:
        """
        Compute the cosine similarity between two users.

        Parameters:
            userU (int): ID of the first user.
            userV (int): ID of the second user.

        Returns:
            float: Cosine similarity score between the two users.
        """
        pass

    @abstractmethod
    def compute_item_similarity(self, itemU: int, itemV: int) -> float:
        """
        Compute the cosine similarity between two items.

        Parameters:
            itemU (int): ID of the first item.
            itemV (int): ID of the second item.

        Returns:
            float: Cosine similarity score between the two items.
        """
        pass


class CosineSimilarity(Similarity):
    """
    Computes the cosine similarity between items.
    """
    def compute_user_similarity(self, userU: int, userV: int) -> float:
        """
        Compute the cosine similarity between two users.

        Parameters:
            userU (int): ID of the first user.
            userV (int): ID of the second user.

        Returns:
            float: Cosine similarity score between the two users.
        """
        # Obtain the ratings of the users
        userU_ratings = self.data.get_interaction_from_user(userU)
        userV_ratings = self.data.get_interaction_from_user(userV)

        # Obtain the common items rated by both users
        # I_uv = {i ∈ I_u ∩ I_v}
        I_uv = set(userU_ratings.keys()).intersection(userV_ratings.keys())

        # If there are no common items, return 0
        if not I_uv:
            return 0.0

        # DUDA 3. Ver si puedo hacer esto con numpy
        # U = {r_ui | i ∈ I_u}
        U = np.array([userU_ratings[item] for item in I_uv])
        # V = {r_vi | i ∈ I_v}
        V = np.array([userV_ratings[item] for item in I_uv])

        # Calculate the dot product and magnitudes
        # dp = Σ_(i∈I_uv) r_ui * r_vi
        dot_product = np.dot(U, V)
        # ||U|| = sqrt(Σ_(i∈I_u) r_ui^2)
        magnitude_U = np.linalg.norm(U)
        # ||V|| = sqrt(Σ_(i∈I_v) r_vi^2)
        magnitude_V = np.linalg.norm(V)

        # Calculate the cosine similarity
        if magnitude_U == 0 or magnitude_V == 0:
            return 0.0
        else:
            # CV(u, v) = cos(x_u, x_v) = dp / (||U|| * ||V||)
            return dot_product / (magnitude_U * magnitude_V)

    def compute_item_similarity(self, itemU, itemV):
        """
        Compute the cosine similarity between two items.

        Parameters:
            itemU (int): ID of the first item.
            itemV (int): ID of the second item.

        Returns:
            float: Cosine similarity score between the two items.
        """
        # Obtain the ratings of the items
        itemU_ratings = self.data.get_interaction_from_item(itemU)
        itemV_ratings = self.data.get_interaction_from_item(itemV)

        # Obtain the common users who rated both items
        # I_uv = {u ∈ U_u ∩ U_v}
        I_uv = set(itemU_ratings.keys()).intersection(itemV_ratings.keys())

        # If there are no common users, return 0
        if not I_uv:
            return 0.0

        # DUDA 3. Ver si puedo hacer esto con numpy
        # U = {r_ui | u ∈ U_u}
        U = np.array([itemU_ratings[user] for user in I_uv])
        # V = {r_vi | u ∈ U_v}
        V = np.array([itemV_ratings[user] for user in I_uv])

        # Calculate the dot product and magnitudes
        # dp = Σ_(u∈I_uv) r_ui * r_vi
        dot_product = np.dot(U, V)
        # ||U|| = sqrt(Σ_(u∈U_u) r_ui^2)
        magnitude_U = np.linalg.norm(U)
        # ||V|| = sqrt(Σ_(u∈U_v) r_vi^2)
        magnitude_V = np.linalg.norm(V)

        # Calculate the cosine similarity
        if magnitude_U == 0 or magnitude_V == 0:
            return 0.0
        else:
            # CV(u, v) = cos(x_u, x_v) = dp / (||U|| * ||V||)
            return dot_product / (magnitude_U * magnitude_V)


class PearsonCorrelationSimilarity(Similarity):
    """
    Computes the Pearson correlation similarity between items.
    """

    @staticmethod
    def _pearson(arr1: np.ndarray, arr2: np.ndarray) -> float:
        """
        Compute the Pearson correlation coefficient between two arrays.

        Parameters:
            arr1 (np.ndarray): First array.
            arr2 (np.ndarray): Second array.

        Returns:
            float: Pearson correlation coefficient between the two arrays.
        """
        # Calculate the mean of each array
        mean1 = np.mean(arr1)
        mean2 = np.mean(arr2)

        # Calculate the covariance and standard deviations
        # numerator = cov(U, V) = Σ_(i∈I_uv) (r_ui - mean_U) * (r_vi - mean_V)
        numerator = np.sum((arr1 - mean1) * (arr2 - mean2))
        # denominator = σ_U * σ_V
        # σ_U = sqrt(Σ_(i∈I_u) (r_ui - mean_U)^2)
        # σ_V = sqrt(Σ_(i∈I_v) (r_vi - mean_V)^2)
        denominator = np.sqrt(np.sum((arr1 - mean1)**2) * np.sum((arr2 - mean2)**2))
        if denominator == 0:
            return 0.0

        # Calculate the Pearson correlation similarity
        # PC(u, v) = cov(U, V) / (σ_U * σ_V) = numerator / denominator
        return numerator / denominator

    def compute_user_similarity(self, userU: int, userV: int) -> float:
        """
        Compute the Pearson correlation similarity between two users.

        Parameters:
            userU (int): ID of the first user.
            userV (int): ID of the second user.

        Returns:
            float: Pearson correlation similarity score between the two users.
        """
        # Obtain the ratings of the users
        userU_ratings = self.data.get_interaction_from_user(userU)
        userV_ratings = self.data.get_interaction_from_user(userV)

        # Obtain the common items rated by both users
        # I_uv = {i ∈ I_u ∩ I_v}
        I_uv = set(userU_ratings.keys()).intersection(userV_ratings.keys())

        # If there are no common items, return 0
        if not I_uv:
            return 0.0

        # DUDA 3. Ver si puedo hacer esto con numpy
        # U = {r_ui | i ∈ I_u}
        U = np.array([userU_ratings[item] for item in I_uv])
        # V = {r_vi | i ∈ I_v}
        V = np.array([userV_ratings[item] for item in I_uv])

        # Calculate the Pearson correlation similarity
        return self._pearson(U, V)

    def compute_item_similarity(self, itemU, itemV):
        """
        Compute the Pearson correlation similarity between two items.

        Parameters:
            itemU (int): ID of the first item.
            itemV (int): ID of the second item.

        Returns:
            float: Pearson correlation similarity score between the two items.
        """
        # Obtain the ratings of the items
        itemU_ratings = self.data.get_interaction_from_item(itemU)
        itemV_ratings = self.data.get_interaction_from_item(itemV)

        # Obtain the common users who rated both items
        # I_uv = {u ∈ U_u ∩ U_v}
        I_uv = set(itemU_ratings.keys()).intersection(itemV_ratings.keys())

        # If there are no common users, return 0
        if not I_uv:
            return 0.0

        # DUDA 3. Ver si puedo hacer esto con numpy
        # U = {r_ui | u ∈ U_u}
        U = np.array([itemU_ratings[user] for user in I_uv])
        # V = {r_vi | u ∈ U_v}
        V = np.array([itemV_ratings[user] for user in I_uv])

        # Calculate the Pearson correlation similarity
        return self._pearson(U, V)
