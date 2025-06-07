import random
from torch.utils.data import Dataset


class BPRDataset(Dataset):
    """
    BPR-MF Dataset that generates triplets (user, positive_item, negative_item).

    Attributes:
        interactions (list of tuple): List of (u, i) positive interactions.
        user_pos (dict): Mapping from user to set of positive items.
        n_items (int): Total number of items (for negative sampling).
    """
    def __init__(self, train_coo, n_items):
        """
        Initialize the BPRDataset.

        Parameters:
            train_coo (scipy.sparse.coo_matrix): Training sparse matrix in COO format
                containing internal user indices (row), internal item indices (col), and ratings.
            n_items (int): Total number of items in the dataset.
        """
        # Store the number of items
        self.n_items = n_items

        # Convert rows and columns to a list of (user, item) tuples
        self.interactions = list(zip(train_coo.row, train_coo.col))

        # Build a dictionary for quick access to positive items per user
        self.user_pos = {}
        for u, i in self.interactions:
            self.user_pos.setdefault(u, set()).add(i)

    def __len__(self):
        """
        Return the number of positive interactions.

        Returns:
            int: Number of positive user-item pairs.
        """
        return len(self.interactions)

    def __getitem__(self, idx):
        """
        Return a triplet (u, i, j) where:
          - i is a positive item for user u.
          - j is a negative item (not in user_pos[u]), sampled at random.

        Parameters:
            idx (int): Index of the positive interaction in the list.

        Returns:
            tuple: (user_index, positive_item_index, negative_item_index)
        """
        u, pos_i = self.interactions[idx]

        # Sample a negative item until it is not in the user's positive set
        neg_j = random.randrange(self.n_items)
        while neg_j in self.user_pos[u]:
            neg_j = random.randrange(self.n_items)

        return u, pos_i, neg_j
