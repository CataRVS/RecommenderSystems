from abc import ABC, abstractmethod
from src.datamodule.data import Data
import numpy as np


class Similarity(ABC):
    """
    Abstract base class for similarity measures.

    Attributes:
        data (Data): Data instance with the user-item interactions.
        sim_matrix (np.ndarray): Precomputed similarity matrix for all users/items.
    """
    def __init__(self, data: Data):
        """
        Create a new Similarity instance.

        Parameters:
            data (Data): Data instance with the user-item interactions.
        """
        self.data = data
        self.sim_matrix = None

    @abstractmethod
    def compute_similarity(self, elemU: int, elemV: int) -> float:
        """
        Compute the cosine similarity between two elements.

        Parameters:
            elemU (int): ID of the first element.
            elemV (int): ID of the second element.

        Returns:
            float: Cosine similarity score between the two elements.
        """
        pass


class CosineSimilarityUsers(Similarity):
    """
    Computes the cosine similarity between users. We will precompute the similarity
    matrix for all users and return the similarity score for the given users.

    Attributes:
        data (Data): Data instance with the user-item interactions.
        sim_matrix (np.ndarray): Precomputed cosine similarity matrix for all users.
    """
    def __init__(self, data: Data):
        """
        Create a new CosineSimilarity instance for users. It computes the cosine
        similarity matrix for all users based on the user-item interactions.

        Parameters:
            data (Data): Data instance with the user-item interactions.
        """
        # Call the parent constructor
        super().__init__(data)

        # To hurry up the computation, we will use the sparse matrix representation
        M = data.get_train_sparse_matrix()

        # Compute the cosine similarity matrix for all users
        # First, we multiply the matrix by its transpose to get the dot product of all users
        dot_product = M.dot(M.T).toarray() # Users x Users

        # Then, we compute the norms of each user to normalize the dot product
        # The norms are the square root of the sum of squares of each row
        # So be first multiply the matrix by itself and then sum the rows
        # A1 is used to convert the matrix to a 1D array view of the matrix
        # We finally take the square root to get the norms (user = dim 0 -> sum axis=1)
        norms = np.sqrt(M.multiply(M).sum(axis=1).A1)

        # Finally, we normalize the dot product by the norms to get the cosine similarity
        # We use the outer product of the norms to get the denominator
        # The outer product is a matrix where each element is the product of the
        # corresponding elements of the two vectors (the product of the norms of the 2
        # users)
        denominator = np.outer(norms, norms)
        # We save the positions where the denominator is 0 to avoid division by 0
        zero_mask = (denominator == 0)
        # We then set the denominator to 1 to avoid division by 0 in those positions
        denominator[zero_mask] = 1
        # Then, we compute the cosine similarity matrix by dividing the dot product
        # by the denominator
        similarity_matrix = dot_product / denominator
        # Finally, we change back the positions where the denom was 0 to 0
        similarity_matrix[zero_mask] = 0

        # Once we have the similarity matrix, we save it
        self.sim_matrix = similarity_matrix

    def compute_similarity(self, userU: int, userV: int) -> float:
        """
        Return the cosine similarity between two users from the precomputed similarity
        matrix.

        Parameters:
            userU (int): ID of the first user.
            userV (int): ID of the second user.

        Returns:
            float: Cosine similarity score between the two users.
        """
        return float(self.sim_matrix[userU, userV])


class CosineSimilarityItems(Similarity):
    """
    Computes the cosine similarity between items. We will precompute the similarity
    matrix for all items and return the similarity score for the given items.

    Attributes:
        data (Data): Data instance with the user-item interactions.
        sim_matrix (np.ndarray): Precomputed cosine similarity matrix for all items.
    """
    def __init__(self, data: Data):
        """
        Create a new CosineSimilarity instance for items. It computes the cosine
        similarity matrix for all items based on the user-item interactions.

        Parameters:
            data (Data): Data instance with the user-item interactions.
        """
        # Call the parent constructor
        super().__init__(data)

        # To hurry up the computation, we will use the sparse matrix representation transposed
        M_t = data.get_train_sparse_matrix().T

        # Compute the cosine similarity matrix for all items
        # First, we multiply the matrix transpose by the matrix to get the dot product
        # of all items.
        dot_product = M_t.dot(M_t.T).toarray()  # Items x Items

        # Then, we compute the norms of each item to normalize the dot product
        # The norms are the square root of the sum of squares of each row
        # So be first multiply the matrix by itself and then sum the rows
        # A1 is used to convert the matrix to a 1D array view of the matrix
        # We finally take the square root to get the norms (item = dim 0 -> sum axis=1)
        # We use the transpose to get the norms for each item
        norms = np.sqrt(M_t.multiply(M_t).sum(axis=1).A1)

        # Finally, we normalize the dot product by the norms to get the cosine similarity
        # We use the outer product of the norms to get the denominator
        # The outer product is a matrix where each element is the product of the
        # corresponding elements of the two vectors (the product of the norms of the 2
        # items)
        denominator = np.outer(norms, norms)
        # We save the positions where the denominator is 0 to avoid division by 0
        zero_mask = (denominator == 0)
        # We then set the denominator to 1 to avoid division by 0 in those positions
        denominator[zero_mask] = 1
        # Then, we compute the cosine similarity matrix by dividing the dot product by
        # the denominator
        similarity_matrix = dot_product / denominator
        # Finally, we change back the positions where the denom was 0 to 0
        similarity_matrix[zero_mask] = 0

        # Once we have the similarity matrix, we save it
        self.sim_matrix = similarity_matrix

    def compute_similarity(self, itemU: int, itemV: int) -> float:
        """
        Return the cosine similarity between two items from the precomputed similarity
        matrix.

        Parameters:
            itemU (int): ID of the first item.
            itemV (int): ID of the second item.

        Returns:
            float: Cosine similarity score between the two items.
        """
        return float(self.sim_matrix[itemU, itemV])


class PearsonCorrelationUsers(Similarity):
    """
    Computes the Pearson correlation similarity between users. We will precompute the
    similarity matrix for all users and return the similarity score for the given
    users.

    Attributes:
        data (Data): Data instance with the user-item interactions.
        sim_matrix (np.ndarray): Precomputed Pearson correlation similarity matrix for
            all users.
    """

    def __init__(self, data: Data):
        super().__init__(data)

        # To hurry up the computation, we will use the sparse matrix representation
        M = data.get_train_sparse_matrix()

        # Compute the Pearson correlation similarity matrix for all users
        # First, we mask the ratings so that we have 1 if the rating is > 0 and 0 otherwise
        # We will use this mask to count the number of ratings for each user
        mask = (M > 0).astype(np.float32)
        

        # We compute the rating mean for each user
        sums = M.sum(axis=1).A1         # Σ r_{u,i}
        counts = mask.sum(axis=1).A1    # |I_u|
        means = sums / counts           # μ_u = Σ r_{u,i} / |I_u|

        # We transform the matrix to coo (coordinate) format to get the row and column indices
        M_coo = M.tocoo()
        M_coo.data = M_coo.data.astype(np.float64) # Ensure data is in float64 for precision
        # We then subtract the mean from each rating to get the centered ratings
        M_coo.data -= means[M_coo.row]
        # We convert it back to csr (compressed sparse row) format to get the dot product
        M_centered = M_coo.tocsr()

        # We compute the dot product of the centered ratings to get the covariance
        dot_product = M_centered.dot(M_centered.T).toarray()  # Users x Users

        # We compute the norms of each user to normalize the dot product
        # The norms are the square root of the sum of squares of each row
        # So be first multiply the matrix by itself and then sum the rows
        # A1 is used to convert the matrix to a 1D array view of the matrix
        # We finally take the square root to get the norms (user = dim 0 -> sum axis=1)
        std = np.sqrt(M_centered.multiply(M_centered).sum(axis=1).A1)

        # We use the outer product of the std to get the denominator
        # The outer product is a matrix where each element is the product of the
        # corresponding elements of the two vectors (the product of the std of the 2
        # users)
        denominator = np.outer(std, std)

        # We save the positions where the denominator is 0 to avoid division by 0
        zero_mask = (denominator == 0)
        # We then set the denominator to 1 to avoid division by 0 in those positions
        denominator[zero_mask] = 1
        # Then, we compute the Pearson correlation similarity matrix by dividing the
        # dot product by the denominator
        similarity_matrix = dot_product / denominator
        # Finally, we change back the positions where the denom was 0 to 0
        similarity_matrix[zero_mask] = 0

        # We save the similarity matrix
        self.sim_matrix = similarity_matrix

    def compute_similarity(self, userU: int, userV: int) -> float:
        """
        Return the Pearson correlation similarity between two users from the
        precomputed similarity matrix.

        Parameters:
            userU (int): ID of the first user.
            userV (int): ID of the second user.

        Returns:
            float: Pearson correlation similarity score between the two users.
        """
        return float(self.sim_matrix[userU, userV])


class PearsonCorrelationItems(Similarity):
    """
    Computes the Pearson correlation similarity between items. We will precompute the
    similarity matrix for all items and return the similarity score for the given
    items.

    Attributes:
        data (Data): Data instance with the user-item interactions.
        sim_matrix (np.ndarray): Precomputed Pearson correlation similarity matrix for
            all items.
    """
    def __init__(self, data: Data):
        """
        Create a new PearsonCorrelationSimilarity instance for items. It computes the
        Pearson correlation similarity matrix for all items based on the user-item
        interactions.

        Parameters:
            data (Data): Data instance with the user-item interactions.
        """
        super().__init__(data)

        # To hurry up the computation, we will use the sparse matrix representation transposed
        M_t = data.get_train_sparse_matrix().T

        # Compute the Pearson correlation similarity matrix for all items
        # First, we mask the ratings so that we have 1 if the rating is > 0 and 0 otherwise
        mask = (M_t > 0).astype(np.float32)

        # We compute the rating mean for each item
        sums = M_t.sum(axis=1).A1       # Σ r_{i,u}
        counts = mask.sum(axis=1).A1    # |U_i|
        means = sums / counts           # μ_i = Σ r_{i,u} / |U_i|

        # We transform the matrix to coo (coordinate) format to get the row and column indices
        M_t_coo = M_t.tocoo()
        M_t_coo.data = M_t_coo.data.astype(np.float64) # Ensure data is in float64 for precision

        # We then subtract the mean from each rating to get the centered ratings
        M_t_coo.data -= means[M_t_coo.row]

        # We convert it back to csr (compressed sparse row) format to get the dot product
        M_centered = M_t_coo.tocsr()

        # We compute the dot product of the centered ratings to get the covariance
        dot_product = M_centered.dot(M_centered.T).toarray()  # Items x Items

        # We compute the norms of each item to normalize the dot product
        # The norms are the square root of the sum of squares of each row
        # So be first multiply the matrix by itself and then sum the rows
        # A1 is used to convert the matrix to a 1D array view of the matrix
        # We finally take the square root to get the norms (item = dim 0 -> sum axis=1)
        std = np.sqrt(M_centered.multiply(M_centered).sum(axis=1).A1)

        # We use the outer product of the std to get the denominator
        # The outer product is a matrix where each element is the product of the
        # corresponding elements of the two vectors (the product of the std of the 2
        # items)
        denominator = np.outer(std, std)

        # We save the positions where the denominator is 0 to avoid division by 0
        zero_mask = (denominator == 0)
        # We then set the denominator to 1 to avoid division by 0 in those positions
        denominator[zero_mask] = 1
        # Then, we compute the Pearson correlation similarity matrix by dividing the
        # dot product by the denominator
        similarity_matrix = dot_product / denominator
        # Finally, we change back the positions where the denom was 0 to 0
        similarity_matrix[zero_mask] = 0

        # We save the similarity matrix
        self.sim_matrix = similarity_matrix

    def compute_similarity(self, itemU: int, itemV: int) -> float:
        """
        Return the Pearson correlation similarity between two items from the
        precomputed similarity matrix.

        Parameters:
            itemU (int): ID of the first item.
            itemV (int): ID of the second item.

        Returns:
            float: Pearson correlation similarity score between the two items.
        """
        return float(self.sim_matrix[itemU, itemV])
