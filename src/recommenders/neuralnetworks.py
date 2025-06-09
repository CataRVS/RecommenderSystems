import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from src.recommenders.basic_recommenders import Recommender
from src.datamodule.data import Data
from src.utils.utils import Recommendation
from src.utils.strategies import Strategy


class MLPRecommender(Recommender):
    """
    Neural Network Recommender using a simple MLP on user and item embeddings.

    Architecture:
        - user embedding of size [n_users, embedding_dim]
        - item embedding of size [n_items, embedding_dim]
        - MLP: linear -> ReLU -> ... -> linear -> output
    Trained with MSE loss on observed ratings.

    Args:
        data (Data): instance holding train/test splits and mappings.
        embedding_dim (int): dimension of user/item embeddings.
        hidden_dims (List[int]): sizes of hidden layers in the MLP.
        lr (float): learning rate for AdamW optimizer.
        weight_decay (float): L2 regularization coefficient.
        n_epochs (int): number of training epochs.
        batch_size (int): mini-batch size.
        device (str|None): 'cpu' or 'cuda', or None for auto-detect.
    """
    def __init__(
        self,
        data: Data,
        embedding_dim: int = 32,
        hidden_dims: list[int] = [64, 32],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 20,
        batch_size: int = 4096,
        device: str | None = None,
    ):
        super().__init__(data)
        # Save dataset and device
        self.data = data
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Dimensions
        n_users = data.get_total_users()
        n_items = data.get_total_items()
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims

        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, 0.0, 0.1)
        nn.init.normal_(self.item_embedding.weight, 0.0, 0.1)

        # Build MLP with Dropout
        layers = []
        input_dim = 2 * embedding_dim
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))  # final score
        self.mlp = nn.Sequential(*layers)

        # Move modules to device
        self.user_embedding.to(self.device)
        self.item_embedding.to(self.device)
        self.mlp.to(self.device)

        # Optimizer and loss
        self.optimizer = torch.optim.AdamW(
            list(self.user_embedding.parameters())
            + list(self.item_embedding.parameters())
            + list(self.mlp.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()

        # Train
        self.user_embedding_np, self.item_embedding_np, self.mlp = \
            self._train(
                n_epochs=n_epochs,
                batch_size=batch_size
            )

    def _train(self, n_epochs: int, batch_size: int):
        """
        Train the MLP on observed (user, item, rating) tuples.

        Returns:
            user_embedding_np (np.ndarray): [n_users, embedding_dim]
            item_embedding_np (np.ndarray): [n_items, embedding_dim]
            mlp (nn.Module): trained MLP (on device in eval mode).
        """
        # Obtain training data
        users_data, items_data, ratings_data = self.data.get_interactions()
        users = torch.tensor(users_data, dtype=torch.long)
        items = torch.tensor(items_data, dtype=torch.long)
        ratings = torch.tensor(ratings_data, dtype=torch.float32)

        dataset = TensorDataset(users, items, ratings)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.user_embedding.train()
        self.item_embedding.train()
        self.mlp.train()

        for epoch in tqdm(range(1, n_epochs + 1), desc="Training MLP"):
            epoch_loss = 0.0
            count = 0
            for u_batch, i_batch, r_batch in loader:
                u_batch = u_batch.to(self.device)
                i_batch = i_batch.to(self.device)
                r_batch = r_batch.to(self.device)

                # Forward pass
                u_emb = self.user_embedding(u_batch)
                i_emb = self.item_embedding(i_batch)
                x = torch.cat([u_emb, i_emb], dim=1)  # [B, 2*emb]
                preds = self.mlp(x).squeeze(1)        # [B]

                # Compute MSE loss
                loss = self.criterion(preds, r_batch)

                # Backward and step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * u_batch.size(0)
                count += u_batch.size(0)

            avg_loss = epoch_loss / count
            tqdm.write(f"Epoch {epoch}/{n_epochs}, Loss: {avg_loss:.4f}")

        # After training, detach embeddings
        user_embedding_np = self.user_embedding.weight.detach().cpu().numpy()
        item_embedding_np = self.item_embedding.weight.detach().cpu().numpy()

        # Keep MLP in eval mode on device for inference
        self.mlp.eval()

        return user_embedding_np, item_embedding_np, self.mlp

    def recommend(
        self,
        user_id: int,
        strategy: Strategy,
        recommendation: Recommendation | None = None,
        n: int = 10
    ) -> Recommendation:
        """
        Generate top-n recommendations for a given user via the trained MLP.
        """
        if recommendation is None:
            recommendation = Recommendation()

        # Filter candidate items
        candidates = strategy.filter(user_id)
        u_idx = self.data.to_internal_user(user_id)
        if u_idx is None or not candidates:
            return recommendation

        # Get user embedding from numpy cache
        u_vec = self.user_embedding_np[u_idx]                   # [emb]
        cand_idxs = [self.data.to_internal_item(c) for c in candidates]
        valid_pairs = [(c, idx) for c, idx in zip(candidates, cand_idxs) if idx is not None]
        if not valid_pairs:
            return recommendation

        items_list, idx_list = zip(*valid_pairs)
        idx_arr = np.array(idx_list, dtype=np.int64)

        # Convert to torch for MLP forward (batch inference)
        u_rep = torch.from_numpy(u_vec).to(self.device)                            # [emb]
        i_rep = torch.from_numpy(self.item_embedding_np[idx_arr]).to(self.device)  # [len, emb]

        # Repeat user embedding and concat
        u_batch = u_rep.unsqueeze(0).repeat(i_rep.size(0), 1)  # [len, emb]
        x = torch.cat([u_batch, i_rep], dim=1)                 # [len, 2*emb]

        # Score via MLP
        with torch.no_grad():
            scores = self.mlp(x).squeeze(1).cpu().numpy()     # [len]

        # Select top-n
        top_k = min(n, scores.shape[0])
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_sorted = top_idx[np.argsort(-scores[top_idx])]

        recs = [(items_list[i], float(scores[i])) for i in top_sorted]
        recommendation.add_recommendations(user_id, recs)
        return recommendation


class GNNRecommender(Recommender):
    """
    Graph Neural Network Recommender using a GCN on user and item embeddings.

    At
    """
    def __init__(
        self,
        data: Data,
        embedding_dim: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 20,
        batch_size: int = 4096,
        n_layers: int = 3,
        device: str | None = None,
    ):
        super().__init__(data)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        n_users = data.get_total_users()
        n_items = data.get_total_items()
        self.n_nodes = n_users + n_items
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_layers = n_layers

        # 1) Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, 0.0, 0.1)
        nn.init.normal_(self.item_embedding.weight, 0.0, 0.1)

        # 2) Build normalized adjacency once
        users_data, items_data, _ = self.data.get_interactions()
        # bipartite edges: user → (n_users + item)
        rows = torch.tensor(np.concatenate([users_data, items_data + n_users]), dtype=torch.long)
        cols = torch.tensor(np.concatenate([items_data + n_users, users_data]), dtype=torch.long)
        idx = torch.stack([rows, cols], dim=0)
        # compute symmetric degree
        deg = np.bincount(idx[0].cpu().numpy(), minlength=self.n_nodes)
        deg_inv_sqrt = torch.tensor(1.0 / np.sqrt(deg + 1e-12), dtype=torch.float32)
        # edge weights = 1/sqrt(deg[u]*deg[v])
        vals = deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]
        self.adj = torch.sparse_coo_tensor(
            idx, vals, (self.n_nodes, self.n_nodes)
        ).coalesce().to(self.device)

        # 3) Optimizer & loss
        self.optimizer = torch.optim.AdamW(
            list(self.user_embedding.parameters())
            + list(self.item_embedding.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()

        # 4) Train
        self.user_final, self.item_final = self._train()

    def _propagate(self, all_embeddings: torch.Tensor) -> torch.Tensor:
        """Propagate through K GCN layers and average."""
        all_embeddings = all_embeddings.to(self.device)  # [N, D]
        embeds = [all_embeddings]
        h = all_embeddings
        for _ in range(self.n_layers):
            # sparse mm
            h = torch.sparse.mm(self.adj, h)
            embeds.append(h)
        # mean of [E⁽⁰⁾…E⁽ᴷ⁾]
        return torch.stack(embeds, dim=1).mean(dim=1)

    def _train(self):
        # prepare training data
        users_data, items_data, ratings_data = self.data.get_interactions()
        users = torch.tensor(users_data, dtype=torch.long)
        items = torch.tensor(items_data, dtype=torch.long)
        ratings = torch.tensor(ratings_data, dtype=torch.float32)
        ds = TensorDataset(users, items, ratings)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        n_users = self.data.get_total_users()

        self.user_embedding.train()
        self.item_embedding.train()

        for epoch in range(1, self.n_epochs + 1):
            epoch_loss = 0.0
            count = 0

            # propagate once per epoch
            E0 = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

            for u_batch, i_batch, r_batch in tqdm(loader, desc=f"GNN Epoch {epoch}", unit="batch"):
                u = u_batch.to(self.device)
                i = (i_batch + n_users).to(self.device)
                r = r_batch.to(self.device)

                # look up embeddings
                E_final = self._propagate(E0)
                u_emb = E_final[u]  # [B, D]
                i_emb = E_final[i]  # [B, D]

                preds = (u_emb * i_emb).sum(dim=1)

                loss = self.criterion(preds, r)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * u.size(0)
                count += u.size(0)

            print(f"Epoch {epoch}/{self.n_epochs}, Loss: {epoch_loss/count:.4f}")

        # after training, compute final embeddings for inference
        E0 = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        E_final = self._propagate(E0).detach().cpu().numpy()
        # split back into user/item
        return E_final[: self.data.get_total_users()], E_final[self.data.get_total_users():]

    def recommend(
        self,
        user_id: int,
        strategy: Strategy,
        recommendation: Recommendation | None = None,
        n: int = 10
    ) -> Recommendation:
        if recommendation is None:
            recommendation = Recommendation()

        candidates = strategy.filter(user_id)
        u_idx = self.data.to_internal_user(user_id)
        if u_idx is None or not candidates:
            return recommendation

        u_emb = self.user_final[u_idx]  # [D]
        mapped = [(c, self.data.to_internal_item(c)) for c in candidates]
        valid = [(c, idx) for c, idx in mapped if idx is not None]
        if not valid:
            return recommendation

        items, idxs = zip(*valid)
        idxs = np.array(idxs, dtype=np.int64)
        v_embs = self.item_final[idxs]   # [M, D]
        scores = np.dot(v_embs, u_emb)   # [M]

        top_k = min(n, len(scores))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_sorted = top_idx[np.argsort(-scores[top_idx])]

        recs = [(items[i], float(scores[i])) for i in top_sorted]
        recommendation.add_recommendations(user_id, recs)
        return recommendation
