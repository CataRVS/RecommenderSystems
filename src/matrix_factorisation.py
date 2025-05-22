import numpy as np
import random
from src.recommenders import Recommender
from src.utils import Recommendation
from src.data import Data
from src.strategies import Strategy

class MFRecommender(Recommender):
    """
    Simple Matrix Factorization Recommender using SGD to minimize squared error.

    This implementation uses a user-item matrix factorization approach.
    It learns latent factors for users and items, and uses these factors to predict ratings.
    The model is trained using stochastic gradient descent (SGD) to minimize the squared error
    between the predicted ratings and the actual ratings in the training data.

    Attributes:
        data (Data): The dataset containing user-item interactions.
        n_users (int): Number of users in the dataset.
        n_items (int): Number of items in the dataset.
        n_factors (int): Number of latent factors for users and items.
        lr (float): Learning rate for SGD.
        regularization (float): Regularization parameter to prevent overfitting.
        n_epochs (int): Number of epochs for training.
        user_factors (np.ndarray): Latent factors for users.
        item_factors (np.ndarray): Latent factors for items.
    """
    def __init__(
        self,
        data: Data,
        n_factors: int = 20,
        lr: float = 0.01,
        regularization: float = 0.1,
        n_epochs: int = 10
    ):
        super().__init__(data)
        self.n_users = data.get_total_users()
        self.n_items = data.get_total_items()
        self.n_factors = n_factors
        self.lr = lr
        self.regularization = regularization
        self.n_epochs = n_epochs
        # We initialize the user and item latent factors with a normal distribution
        # with mean 0 and standard deviation 0.1.
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, n_factors))
        # We train the model using stochastic gradient descent (SGD).
        self._train_sgd()

    def _train_sgd(self):
        # We get the training data as a sparse matrix in COO format.
        # The COO format is efficient for constructing sparse matrices.
        train_mat = self.data.get_train_sparse_matrix().tocoo()
        users, items, ratings = train_mat.row, train_mat.col, train_mat.data

        # For each epoch:
        for _ in range(self.n_epochs):
            # We shuffle the indices of the training data to ensure that the SGD updates are random.
            idx = np.arange(len(ratings))
            np.random.shuffle(idx)
            # For each user-item-rating triplet in the training data:
            for k in idx:
                u, i, r_ui = users[k], items[k], ratings[k]
                # We predict the rating using the dot product of the user and item latent factors.
                pred = self.user_factors[u].dot(self.item_factors[i])
                # We compute the error as the difference between the actual rating and the predicted rating.
                # We update the user and item latent factors using the SGD update rule.
                err = r_ui - pred
                u_vec = self.user_factors[u]
                i_vec = self.item_factors[i]
                # We update the user and item latent factors using the SGD update rule.
                # The update rule includes a regularization term to prevent overfitting.
                self.user_factors[u] += self.lr * (err * i_vec - self.regularization * u_vec)
                self.item_factors[i] += self.lr * (err * u_vec - self.regularization * i_vec)

    def recommend(
        self,
        user_id: int,
        strategy: Strategy,
        recommendation: Recommendation = None,
        n: int = 10
    ) -> Recommendation:
        if recommendation is None:
            recommendation = Recommendation()
        # Get the candidates for the user using the strategy.
        candidates = strategy.filter(user_id)
        u_idx = self.data.to_internal_user(user_id)
        if u_idx is None:
            return recommendation
        # Get the user latent factors for the user.
        u_vec = self.user_factors[u_idx]
        # Map the candidates to their internal indices.
        # Filter out candidates that are not in the training data.
        mapped = [(cand, self.data.to_internal_item(cand)) for cand in candidates]
        valid = [(cand, idx) for cand, idx in mapped if idx is not None]
        if not valid:
            return recommendation
        # Get the item indices and their corresponding internal indices.
        # Unpack the valid candidates into two lists: items_list and idx_list.
        items_list, idx_list = zip(*valid)
        # Compute the scores for the items using the dot product of the item latent
        # factors and the user latent factors.
        scores = np.dot(self.item_factors[list(idx_list)], u_vec)
        # Get the top n items with the highest scores.
        indices = np.argsort(-scores)

        # If the number of candidates is greater than n, we take the top n items.
        # Otherwise, we take all the candidates.
        if len(indices) > n:
            top_idx = indices[:n]
        else:
            top_idx = indices
        # Create a list of recommendations with the item ID and the corresponding score.
        # The score is converted to a float for better readability.
        recs = [(items_list[i], float(scores[i])) for i in top_idx]
        # Add the recommendations to the recommendation object.
        # The recommendations are added to the recommendation object using the user_id.
        recommendation.add_recommendations(user_id, recs)
        # Return the recommendation object with the recommendations.
        return recommendation

class BPRMFRecommender(Recommender):
    """
    BPR-MF Recommender System using Bayesian Personalized Ranking (BPR) with Matrix Factorization.
    """
    def __init__(
        self,
        data: Data,
        n_factors: int = 20,
        learning_rate: float = 0.01,
        reg: float = 0.01,
        n_iter: int = 100_000
    ):
        super().__init__(data)
        self.n_users = data.get_total_users()
        self.n_items = data.get_total_items()
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = reg
        self.n_iter = n_iter
        # initialize latent factors
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        # build user->positive items map
        train_mat = self.data.get_train_sparse_matrix().tocoo()
        self.user_pos = {u: [] for u in range(self.n_users)}
        for u, i in zip(train_mat.row, train_mat.col):
            self.user_pos[u].append(i)
        # train model
        self._train_bpr()

    def _train_bpr(self):
        for _ in range(self.n_iter):
            u = random.randint(0, self.n_users - 1)
            pos = self.user_pos.get(u)
            if not pos:
                continue
            i = random.choice(pos)
            j = random.randint(0, self.n_items - 1)
            while j in pos:
                j = random.randint(0, self.n_items - 1)
            u_vec = self.user_factors[u]
            i_vec = self.item_factors[i]
            j_vec = self.item_factors[j]
            x_uij = u_vec.dot(i_vec) - u_vec.dot(j_vec)
            sigmoid = 1 / (1 + np.exp(-x_uij))
            # gradient updates
            self.user_factors[u] += self.lr * ((1 - sigmoid) * (i_vec - j_vec) - self.reg * u_vec)
            self.item_factors[i] += self.lr * ((1 - sigmoid) * u_vec - self.reg * i_vec)
            self.item_factors[j] += self.lr * (-(1 - sigmoid) * u_vec - self.reg * j_vec)

    def recommend(
        self,
        user_id: int,
        strategy: Strategy,
        recommendation: Recommendation = None,
        n: int = 10
    ) -> Recommendation:
        if recommendation is None:
            recommendation = Recommendation()
        candidates = strategy.filter(user_id)
        u_idx = self.data.to_internal_user(user_id)
        if u_idx is None:
            return recommendation
        u_vec = self.user_factors[u_idx]
        mapped = [(cand, self.data.to_internal_item(cand)) for cand in candidates]
        valid = [(cand, idx) for cand, idx in mapped if idx is not None]
        if not valid:
            return recommendation
        items_list, idx_list = zip(*valid)
        scores = np.dot(self.item_factors[list(idx_list)], u_vec)
        top_idx = np.argsort(-scores)[:n]
        recs = [(items_list[i], float(scores[i])) for i in top_idx]
        recommendation.add_recommendations(user_id, recs)
        return recommendation
