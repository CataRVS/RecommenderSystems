import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.recommenders.basic_recommenders import Recommender
from src.datamodule.data import Data
from src.utils.utils import Recommendation
from src.utils.strategies import Strategy

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
        reg (float): Regularization parameter to prevent overfitting.
        n_epochs (int): Number of epochs for training.
        user_factors (np.ndarray): Latent factors for users.
        item_factors (np.ndarray): Latent factors for items.
    """
    def __init__(
        self,
        data: Data,
        n_factors: int = 20,
        lr: float = 1e-2,
        regularization: float = 1e-4,
        n_epochs: int = 20,
        batch_size: int = 4096,
        device: str | None = None,
    ):
        super().__init__(data)
        self.n_users: int = data.get_total_users()
        self.n_items: int = data.get_total_items()
        self.n_factors = n_factors
        self.lr = lr
        self.reg = regularization
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # If no device is specified, we use the default device (CPU or GPU).
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Our user and item latent factors are represented as embeddings vectors.
        self.user_factors = nn.Embedding(self.n_users, n_factors)
        self.item_factors = nn.Embedding(self.n_items, n_factors)

        # We initialize them with a normal distribution ~ N(0, 0.1)
        nn.init.normal_(self.user_factors.weight, 0, 0.1)
        nn.init.normal_(self.item_factors.weight, 0, 0.1)

        # We pass them to the chosen device
        self.user_factors.to(self.device)
        self.item_factors.to(self.device)

        # We create the optimizer for the user and item factors.
        self.optimizer = torch.optim.SGD(
            list(self.user_factors.parameters()) + list(self.item_factors.parameters()),
            lr=self.lr,
            weight_decay=self.reg,
        )
        # We train the model using stochastic gradient descent (SGD).
        self._train_sgd()

        # Finally, we move the user and item factors to the cpu and convert them to numpy arrays.
        # This is done to save memory and to make it easier to use them in the recommend method.
        self.user_factors_np = self.user_factors.weight.detach().cpu().numpy()
        self.item_factors_np = self.item_factors.weight.detach().cpu().numpy()

    def _train_sgd(self):
        """
        Train the model using stochastic gradient descent (SGD) to minimize the squared
        error between the predicted ratings and the actual ratings in the training data.
        The model learns latent factors for users and items, and uses these factors to
        predict ratings.

        The training data is loaded in batches using a DataLoader, and the model is
        trained for a specified number of epochs. The loss is computed using mean squared
        error (MSE) between the predicted ratings and the actual ratings. The model
        parameters are updated using the optimizer.
        The training process is displayed using a progress bar.

        """
        # Get the training data as a sparse matrix in COO format to get the
        # user, item and rating tensors.
        train_mat = self.data.get_train_sparse_matrix().tocoo()
        users_t = torch.as_tensor(train_mat.row, dtype=torch.long)
        items_t = torch.as_tensor(train_mat.col, dtype=torch.long)
        ratings_t = torch.as_tensor(train_mat.data, dtype=torch.float32)
        # Create a TensorDataset and DataLoader to handle the training data.
        dataset = TensorDataset(users_t, items_t, ratings_t)
        # Create a DataLoader to iterate over the dataset in batches.
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,  # barajamos interacciones; también se podría barajar usuarios
            drop_last=False
        )
        # Set the model to training mode.
        self.user_factors.train()
        self.item_factors.train()
        # Iterate over the number of epochs.
        for epoch in tqdm(range(self.n_epochs), desc="Training MF", unit="epoch"):
            # Set the epoch loss to 0.
            epoch_loss = 0.0
            # Iterate over the DataLoader to get batches of data.
            for u, i, r in dataloader:
                # Move the data to the specified device (CPU or GPU).
                u = u.to(self.device)
                i = i.to(self.device)
                r = r.to(self.device)

                # The predicted ratings are computed as the dot product of the user and
                # item factors.
                preds = (self.user_factors(u) * self.item_factors(i)).sum(dim=1)

                # Compute the loss using mean squared error (MSE) between the predicted
                # ratings and the actual ratings.
                loss = nn.functional.mse_loss(preds, r)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * u.size(0)

    def recommend(
        self,
        user_id: int,
        strategy: Strategy,
        recommendation: Recommendation | None = None,
        n: int = 10,
    ) -> Recommendation:
        if recommendation is None:
            recommendation = Recommendation()
        # Get the candidates for the user using the strategy.
        candidates = strategy.filter(user_id)
        u_idx = self.data.to_internal_user(user_id)
        if u_idx is None:
            return recommendation
        # Get the user latent factors for the user.
        u_vec = self.user_factors_np[u_idx]
        # Map the candidates to their internal indices.
        # Filter out candidates that are not in the training data.
        mapped = [(cand, self.data.to_internal_item(cand)) for cand in candidates]
        valid = [(cand, idx) for cand, idx in mapped if idx is not None]
        if not valid:
            return recommendation
        # Get the item indices and their corresponding internal indices.
        # Unpack the valid candidates into two lists: items_list and idx_list.
        items_list, idx_list = zip(*valid)

        # Convert the list of indices to a numpy array.
        # This is done to speed up the dot product operation.
        idx_arr = np.fromiter(idx_list, dtype=np.int64)

        # Compute the scores for each candidate item by taking the dot product of the
        # item factors and the user factors.
        scores = self.item_factors_np[idx_arr].dot(u_vec)

        # Get the top n items with the highest scores.
        # We use argpartition to get the indices of the top n items.
        top_idx = np.argpartition(-scores, n - 1)[:n]
        top_sorted = top_idx[np.argsort(-scores[top_idx])]

        # Get the top n items and their corresponding scores.
        # We use a list comprehension to create a list of tuples (item, score).
        recs = [(items_list[i], float(scores[i])) for i in top_sorted]

        # Add the recommendations to the recommendation object and return it.
        recommendation.add_recommendations(user_id, recs)
        return recommendation

class BPRMFRecommender(Recommender):
    """
    BPR-MF Recommender System using Bayesian Personalized Ranking (BPR) with Matrix Factorization.
    """
    