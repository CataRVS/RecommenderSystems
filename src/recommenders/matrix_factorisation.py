import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.recommenders.basic_recommenders import Recommender
from src.datamodule.data import AbstractData
from src.utils.datasets import BPRDataset
from src.utils.utils import Recommendation
from src.utils.strategies import Strategy


class MFRecommender(Recommender):
    """
    Matrix Factorization Recommender: A recommender system based on matrix factorization.

    Attributes:
        data (AbstractData): The dataset containing user-item interactions.
        writer (SummaryWriter): TensorBoard writer for logging training metrics.
        user_factors (np.ndarray): Learned user latent factors after training.
        item_factors (np.ndarray): Learned item latent factors after training.
    """
    def __init__(
        self,
        data: AbstractData,
        embedding_dim: int = 20,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        n_epochs: int = 20,
        batch_size: int = 4096,
        device: str | None = None,
        log_dir: str = "runs/mf_recommender",
    ):
        """
        Initialize the Matrix Factorization Recommender and train the model.

        Parameters:
            data (AbstractData): The dataset containing user-item interactions.
            embedding_dim (int): The dimensionality of the user and item embeddings.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            n_epochs (int): Number of epochs to train the model.
            batch_size (int): Size of the batches for training.
            device (str): Device to run the model on (CPU or GPU).
            log_dir (str): Directory for TensorBoard logs.
        """
        super().__init__(data)
        n_users: int = data.get_total_users()
        n_items: int = data.get_total_items()

        # Create the Writer for TensorBoard logging.
        self.writer = SummaryWriter(log_dir)

        # If no device is specified, we use the default device (CPU or GPU).
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        # Our user and item latent factors are represented as embeddings vectors.
        user_factors = nn.Embedding(n_users, embedding_dim)
        item_factors = nn.Embedding(n_items, embedding_dim)

        # We initialize them with a normal distribution ~ N(0, 0.1)
        nn.init.normal_(user_factors.weight, 0, 0.1)
        nn.init.normal_(item_factors.weight, 0, 0.1)

        # We pass them to the chosen device
        user_factors.to(device)
        item_factors.to(device)

        # We create the optimizer for the user and item factors.
        optimizer = torch.optim.AdamW(
            list(user_factors.parameters()) + list(item_factors.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        # We create the loss function for the user and item factors.
        loss = nn.MSELoss()

        # We train the model using stochastic gradient descent (SGD).
        self.user_factors, self.item_factors = self._train(
            user_factors,
            item_factors,
            optimizer,
            loss,
            n_epochs,
            batch_size,
            device,
        )

        # Close the TensorBoard writer.
        self.writer.close()

    def _train(
        self,
        user_factors: nn.Embedding,
        item_factors: nn.Embedding,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        batch_size: int,
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Train the matrix factorization model using stochastic gradient descent (SGD).

        Parameters:
            user_factors (nn.Embedding): User latent factors.
            item_factors (nn.Embedding): Item latent factors.
            optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
            criterion (nn.Module): Loss function to minimize.
            num_epochs (int): Number of epochs to train the model.
            batch_size (int): Size of the batches for training.
            device (torch.device): Device to run the model on (CPU or GPU).

        Returns:
            tuple: User and item latent factors as numpy arrays after training.
        """
        # Get the training data as a sparse matrix in COO format to get the
        # user, item and rating tensors.
        users, items, ratings = self.data.get_interactions()
        users_t = torch.as_tensor(users, dtype=torch.long)
        items_t = torch.as_tensor(items, dtype=torch.long)
        ratings_t = torch.as_tensor(ratings, dtype=torch.float32)

        # Create a TensorDataset and DataLoader to handle the training data.
        dataset = TensorDataset(users_t, items_t, ratings_t)

        # Create a DataLoader to iterate over the dataset in batches.
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )

        # Set the model to training mode.
        user_factors.train()
        item_factors.train()

        # Iterate over the number of epochs.
        for epoch in tqdm(range(1, num_epochs + 1), desc="MF Training", unit="epoch"):
            # Set the epoch loss to 0.
            epoch_loss = 0.0
            # Iterate over the DataLoader to get batches of data.
            for u, i, r in dataloader:
                # Move the data to the specified device (CPU or GPU).
                u = u.to(device)
                i = i.to(device)
                r = r.to(device)

                # We reset the gradients to zero before the backward pass.
                optimizer.zero_grad()

                # Forward pass of the model: compute the predicted ratings by taking the
                # dot product of the user and item factors.
                pred_rating = (user_factors(u) * item_factors(i)).sum(dim=1)

                # Compute the loss
                loss = criterion(pred_rating, r)

                # Backward pass: compute the gradients.
                loss.backward()

                # Update the model parameters using the optimizer.
                optimizer.step()

                epoch_loss += loss.item() * u.size(0)

            # Log the average loss for the epoch to TensorBoard.
            avg_loss = epoch_loss / len(dataset)
            self.writer.add_scalar("MF_Loss/train", avg_loss, epoch)
            tqdm.write(f"Epoch {epoch} Loss: {avg_loss:.4f}")

        # Print the final loss.
        print(f"Final Loss: {epoch_loss / len(dataset):.4f}")

        # Finally, we move the user and item factors to the cpu and convert them to numpy arrays.
        # This is done to save memory and to make it easier to use them in the recommend method.
        return (
            user_factors.weight.detach().cpu().numpy(),
            item_factors.weight.detach().cpu().numpy(),
        )

    def recommend(
        self,
        user_id: int,
        strategy: Strategy,
        recommendation: Recommendation | None = None,
        n: int = 10,
    ) -> Recommendation:

        # If no recommendation object is passed, we create a new one.
        if recommendation is None:
            recommendation = Recommendation()

        # Get the candidates for the user using the strategy given.
        candidates = strategy.filter(user_id)

        # If no candidates are found, we return the recommendation object.
        if not candidates:
            return recommendation

        # Get the internal user index for the user.
        u_idx = self.data.to_internal_user(user_id)
        # If the user is not in the training data, we return the recommendation object.
        if u_idx is None:
            return recommendation

        # Get the user latent factors
        u_vec = self.user_factors[u_idx]
        # Map the candidates to their internal indices.
        mapped = [(cand, self.data.to_internal_item(cand)) for cand in candidates]
        # Filter out candidates that are not in the training data.
        valid = [(cand, idx) for cand, idx in mapped if idx is not None]
        # If no valid candidates are found, we return the recommendation object.
        if not valid:
            return recommendation
        # Get the item indices and their corresponding internal indices.
        # Unpack the valid candidates into two lists: items_list and idx_list.
        items_list, idx_list = zip(*valid)

        # Convert the list of indices to a numpy array.
        idx_arr = np.fromiter(idx_list, dtype=np.int64)

        # Compute the scores for each candidate item by taking the dot product of the
        # item factors and the user factors.
        scores = self.item_factors[idx_arr].dot(u_vec)

        # If the number of candidates is less than n, we set n to the number of candidates.
        if len(scores) < n:
            n = len(scores)
        # Get the top n items with the highest scores.
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
    BPR-MF Recommender: Matrix Factorization optimized with Bayesian Personalized Ranking.

    Attributes:
        data (AbstractData): The dataset containing user-item interactions.
        writer (SummaryWriter): TensorBoard writer for logging training metrics.
        user_factors (np.ndarray): Learned user latent factors after training.
        item_factors (np.ndarray): Learned item latent factors after training.
    """
    def __init__(
        self,
        data: AbstractData,
        embedding_dim: int = 20,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        n_epochs: int = 20,
        batch_size: int = 4096,
        device: str | None = None,
        log_dir: str = "runs/bprmf_recommender",
    ):
        """
        Initialize the BPR-MF Recommender and train the model.

        Parameters:
            data (AbstractData): The dataset containing user-item interactions.
            embedding_dim (int): The dimensionality of the user and item embeddings.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            n_epochs (int): Number of epochs to train the model.
            batch_size (int): Size of the batches for training.
            device (str): Device to run the model on (CPU or GPU).
            log_dir (str): Directory for TensorBoard logs.
        """
        super().__init__(data)
        n_users = data.get_total_users()
        n_items = data.get_total_items()

        # Create the Writer for TensorBoard logging.
        self.writer = SummaryWriter(log_dir)

        # If no device is specified, we use the default device (CPU or GPU).
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Our user and item latent factors are represented as embeddings vectors.
        user_factors = nn.Embedding(n_users, embedding_dim)
        item_factors = nn.Embedding(n_items, embedding_dim)

        # We initialize them with a normal distribution ~ N(0, 0.1)
        nn.init.normal_(user_factors.weight, 0, 0.1)
        nn.init.normal_(item_factors.weight, 0, 0.1)

        # We pass them to the chosen device
        user_factors.to(device)
        item_factors.to(device)

        # We create the optimizer for the user and item factors.
        optimizer = torch.optim.AdamW(
            list(user_factors.parameters()) + list(item_factors.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        # We train the model using stochastic gradient descent (SGD).
        self.user_factors, self.item_factors = self._train(
            user_factors,
            item_factors,
            optimizer,
            n_epochs,
            batch_size,
            device,
        )

        # Close the TensorBoard writer.
        self.writer.close()

    def _train(
        self,
        user_factors: nn.Embedding,
        item_factors: nn.Embedding,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        batch_size: int,
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Train the BPR-MF model using Bayesian Personalized Ranking (BPR) loss.

        Parameters:
            user_factors (nn.Embedding): User latent factors.
            item_factors (nn.Embedding): Item latent factors.
            optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
            n_epochs (int): Number of epochs to train the model.
            batch_size (int): Size of the batches for training.
            device (torch.device): Device to run the model on (CPU or GPU).

        Returns:
            tuple: User and item latent factors as numpy arrays after training.
        """

        # Create a BPRDataset to handle the training data.
        # This dataset will generate triplets (user, positive_item, negative_item).
        dataset = BPRDataset(self.data)

        # Create a DataLoader to iterate over the dataset in batches.
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )

        for epoch in tqdm(range(1, n_epochs + 1), desc="BPR-MF Training", unit="epoch"):
            epoch_loss = 0.0
            total_triples = 0

            for (u_batch, pos_batch, neg_batch) in dataloader:
                # Move the data to the specified device (CPU or GPU).
                u_batch = u_batch.to(device)        # shape: [B]
                pos_batch = pos_batch.to(device)    # shape: [B]
                neg_batch = neg_batch.to(device)    # shape: [B]

                # Obtain the positive and negative item indices for the users in the batch.
                u_vec = user_factors(u_batch)       # shape: [B, D]
                i_vec = item_factors(pos_batch)     # shape: [B, D]
                j_vec = item_factors(neg_batch)     # shape: [B, D]

                # Forward pass (compute the scores for positive and negative items).
                xui = (u_vec * i_vec).sum(dim=1)    # shape: [B]
                xuj = (u_vec * j_vec).sum(dim=1)    # shape: [B]

                # Compute loss using BPR loss function
                # BPR loss is -log(sigmoid(xui - xuj))
                # We add a small constant to avoid log(0)
                # We do the mean to have the result of the batch
                loss = -torch.log(torch.sigmoid(xui - xuj) + 1e-8).mean()

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * u_batch.size(0)
                total_triples += u_batch.size(0)

            # Log the average loss for the epoch to TensorBoard.
            avg_loss = epoch_loss / total_triples
            self.writer.add_scalar("BPRMF_Loss/train", avg_loss, epoch)
            tqdm.write(f"Epoch {epoch} Loss: {avg_loss:.4f}")

        # Print the final loss for the last epoch.
        print(f"Final Loss: {epoch_loss / total_triples:.4f}")

        # Finally, we move the user and item factors to the cpu and convert them to
        # numpy arrays.
        return (
            user_factors.weight.detach().cpu().numpy(),
            item_factors.weight.detach().cpu().numpy(),
        )

    def recommend(
        self,
        user_id: int,
        strategy: Strategy,
        recommendation: Recommendation | None = None,
        n: int = 10,
    ) -> Recommendation:
        # If no recommendation object is passed, we create a new one.
        if recommendation is None:
            recommendation = Recommendation()

        # Get the candidates for the user using the strategy given.
        candidates = strategy.filter(user_id)
        u_idx = self.data.to_internal_user(user_id)

        # If no candidates are found, we return the recommendation object.
        if u_idx is None:
            return recommendation

        u_vec = self.user_factors[u_idx]
        mapped = [(cand, self.data.to_internal_item(cand)) for cand in candidates]
        valid = [(cand, idx) for cand, idx in mapped if idx is not None]
        if not valid:
            return recommendation

        # Unpack the valid candidates into two lists: items_list and idx_list.
        items_list, idx_list = zip(*valid)
        idx_arr = np.fromiter(idx_list, dtype=np.int64)
        scores = self.item_factors[idx_arr].dot(u_vec)

        # If the number of candidates is less than n, we set n to the number of candidates.
        if len(scores) < n:
            n = len(scores)
        top_idx = np.argpartition(-scores, n - 1)[:n]
        top_sorted = top_idx[np.argsort(-scores[top_idx])]

        # Get the top n items and their corresponding scores.
        recs = [(items_list[i], float(scores[i])) for i in top_sorted]
        # Add the recommendations to the recommendation object and return it.
        recommendation.add_recommendations(user_id, recs)
        # Return the recommendation object.
        return recommendation
