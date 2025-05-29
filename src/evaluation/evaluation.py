from abc import ABC, abstractmethod
from src.datamodule.data import Data
from typing import Dict, List
import numpy as np


class Evaluation(ABC):
    """
    Abstract base class for evaluation metrics.
    """
    def __init__(self, data: Data):
        """
        Create a new Evaluation instance.

        Parameters:
            data (Data): Data instance with the user-item interactions.
        """
        self.data = data

    @abstractmethod
    def evaluate(self, recommendations_path: str) -> float:
        """
        Evaluate the recommendations.

        Parameters:
            recommendations_path (str): Path to the recommendations file.

        Returns:
            float: Evaluation score.
        """
        pass


class Precision(Evaluation):
    """
    Computes the precision of the recommendations.
    """
    def evaluate(
            self,
            recommendations_path: str,
            recommendations_sep: str = ",",
            ignore_first_line: bool = False
        ) -> float:
        """
        Evaluate the recommendations.

        Parameters:
            recommendations_path (str): Path to the recommendations file.
            recommendations_sep (str): Separator used in the recommendations file.
            ignore_first_line (bool): Whether to ignore the first line of the recommendations file.

        Returns:
            float: Precision score.
        """
        # TEST 1: Implement precision evaluation
        # Load the recommendations
        recommendations = self.data._load_recs(
            recommendations_path, recommendations_sep, ignore_first_line
        )

        # Calculate precision for each user
        precisions = []
        for u, items in recommendations.items():
            # Get the test interactions for the user (the ground truth)
            ground_truth = set(self.data.get_test_interactions(u).keys())
            if not items:
                continue
            # Calculate the number of true positives (TP)
            tp = len(set(items) & ground_truth)
            # Calculate precision
            precisions.append(tp / len(items))

        # Return the average precision
        return float(np.mean(precisions)) if precisions else 0.0


class Recall(Evaluation):
    """
    Computes the recall of the recommendations.
    """

    def evaluate(
            self,
            recommendations_path: str,
            recommendations_sep: str = ",",
            ignore_first_line: bool = False
        ) -> float:
        """
        Evaluate the recommendations.

        Parameters:
            recommendations_path (str): Path to the recommendations file.
            recommendations_sep (str): Separator used in the recommendations file.
            ignore_first_line (bool): Whether to ignore the first line of the recommendations file.

        Returns:
            float: Recall score.
        """
        # TEST 2: Implement recall evaluation
        # Load the recommendations
        recommendations = self.data._load_recs(
            recommendations_path, recommendations_sep, ignore_first_line
        )

        # Calculate recall for each user
        recalls = []
        for u, items in recommendations.items():
            # Get the test interactions for the user (the ground truth)
            ground_truth = set(self.data.get_test_interactions(u).keys())
            if not items:
                continue
            # Calculate the number of true positives (TP)
            tp = len(set(items) & ground_truth)
            # Avoid division by zero
            if len(ground_truth) > 0:
                # Calculate recall
                recalls.append(tp / len(ground_truth))
            else:
                # If there are no recalls, return 0.0
                recalls.append(0.0)

        # Return the average recall
        return float(np.mean(recalls)) if recalls else 0.0


class NDCG(Evaluation):
    """
    Computes the normalized discounted cumulative gain (NDCG) of the recommendations.
    """
    def _dcg(self, relevance: list) -> float:
        """
        Compute the DCG for a list of relevance scores.

        Parameters:
            relevance (list): List of relevance scores.

        Returns:
            float: DCG score.
        """
        return sum(r / np.log2(idx+2) for idx, r in enumerate(relevance))

    def evaluate(
            self,
            recommendations_path: str,
            recommendations_sep: str = ",",
            ignore_first_line: bool = False
        ) -> float:
        """
        Evaluate the recommendations.

        Parameters:
            recommendations_path (str): Path to the recommendations file.
            recommendations_sep (str): Separator used in the recommendations file.
            ignore_first_line (bool): Whether to ignore the first line of the recommendations file.

        Returns:
            float: NDCG score.
        """
        # TEST 3: Implement NDCG evaluation
        # Load the recommendations
        recommendations = self.data._load_recs(
            recommendations_path, recommendations_sep, ignore_first_line
        )

        # Calculate NDCG for each user
        ndcgs = []
        for u, items in recommendations.items():
            # Get the test interactions for the user (the ground truth)
            ground_truth = set(self.data.get_test_interactions(u).keys())
            if not items:
                continue
            # Calculate relevance scores
            relevance = [1 if item in ground_truth else 0 for item in items]
            # Calculate DCG
            dcg = self._dcg(relevance)
            # Calculate IDCG (ideal DCG)
            ideal_relevance = [1]*min(len(ground_truth), len(items))
            ideal_dcg = self._dcg(ideal_relevance)
            # Calculate NDCG
            ndcgs.append(dcg / ideal_dcg if ideal_dcg > 0 else 0)

        # Return the average NDCG
        return float(np.mean(ndcgs)) if ndcgs else 0.0


class EPC(Evaluation):
    """
    DUDA 1: Que es EPC?
    Expected Popularity Complement (EPC) of the recommendations.

    EPC@k(u) = (1/k) Σ_{i∈L_{u,k}} (1 - pop(i)) where:
    - L_{u,k} is the list of k recommended items for user u and
    - pop(i) = |UsersWhoConsumed(i)| / |U|.
    """

    def evaluate(
            self,
            recommendations_path: str,
            recommendations_sep: str = ",",
            ignore_first_line: bool = False
        ) -> float:
        """
        Evaluate the recommendations.

        Parameters:
            recommendations_path (str): Path to the recommendations file.
            recommendations_sep (str): Separator used in the recommendations file.
            ignore_first_line (bool): Whether to ignore the first line of the recommendations file.

        Returns:
            float: Expected precision score.
        """
        # TODO 4: Implement expected precision evaluation
        # Load the recommendations
        recommendations: Dict[int, List[int]] = self.data._load_recs(
            recommendations_path, recommendations_sep, ignore_first_line
        )
        # If there are no recommendations, return 0.0
        if not recommendations:
            return 0.0

        # Get the total number of users
        num_users = len(self.data.get_users())
        # If there are no users, return 0.0
        if num_users == 0:
            return 0.0
        
        # Get the total number of items and the number of items in the recommendations
        all_items = {i for lst in recommendations.values() for i in lst}
        pop_cache: Dict[int, float] = {}

        # Calculate the popularity of each item
        for item in all_items:
            # Get the number of users who consumed the item
            num_users_consumed = len(self.data.get_interaction_from_item(item))
            # Calculate the popularity of the item
            pop_cache[item] = num_users_consumed / num_users

        # Calculate EPC for each user
        epc_scores: List[float] = []
        # Iterate over each user and their recommended items
        for u, items in recommendations.items():
            # Get the test interactions for the user (the ground truth)
            if not items:
                epc_scores.append(0.0)
                continue
            # Calculate the complement sum
            # The complement sum is the sum of (1 - pop(i)) for each item i in the recommendations
            complement_sum = sum(1.0 - pop_cache.get(i, 0.0) for i in items)
            # Append the average EPC score for the user
            epc_scores.append(complement_sum / len(items))
        # Return the average EPC score
        return float(np.mean(epc_scores)) if epc_scores else 0.0

class Gini(Evaluation):
    """
    Computes the Gini coefficient of the recommendations.
    """
    def evaluate(
        self,
        recommendations_path: str,
        recommendations_sep: str = ",",
        ignore_first_line: bool = False
    ) -> float:
        """
        Evaluate the recommendations.

        Parameters:
            recommendations_path (str): Path to the recommendations file.
            recommendations_sep (str): Separator used in the recommendations file.
            ignore_first_line (bool): Whether to ignore the first line of the recommendations file.

        Returns:
            float: Gini coefficient score.
        """
        # TEST 5: Implement Gini coefficient evaluation

        # Load the recommendations
        recommendations = self.data._load_recs(
            recommendations_path, recommendations_sep, ignore_first_line
        )
        # Transform the recommendations into a list of items
        items = [item for sublist in recommendations.values() for item in sublist]
        # If there are no items, return 0.0
        if not items:
            return 0.0

        # Count the occurrences of each item
        _, counts = np.unique(items, return_counts=True)

        # Sort the counts in ascending order
        sorted_counts = np.sort(counts)
        # Get the number of items
        n = len(sorted_counts)
        # If there is only one item, return 0.0
        if n == 1:
            return 0.0

        # Get the total number of items
        total = sorted_counts.sum()
        # Calculate the Gini coefficient
        index = np.arange(1, n + 1)
        gini = (2 * (index * sorted_counts).sum()) / (n * total) - (n + 1) / n

        # Return the Gini coefficient
        return float(gini)


class AggregateDiversity(Evaluation):
    """
    Computes the aggregate diversity of the recommendations.
    """
    def evaluate(
            self,
            recommendations_path: str,
            recommendations_sep: str = ",",
            ignore_first_line: bool = False
        ) -> float:
        """
        Evaluate the recommendations.

        Parameters:
            recommendations_path (str): Path to the recommendations file.
            recommendations_sep (str): Separator used in the recommendations file.
            ignore_first_line (bool): Whether to ignore the first line of the recommendations file.

        Returns:
            float: Aggregate diversity score.
        """
        # TEST 6: Implement aggregate diversity evaluation
        # Load the recommendations
        recommendations = self.data._load_recs(
            recommendations_path, recommendations_sep, ignore_first_line
        )

        # Get all items in the recommendations
        all_items = [
            item for sublist in recommendations.values() for item in sublist
        ]
        # Calculate the number of unique items
        num_unique_items = len(set(all_items))
        # Calculate the total number of items recommended
        total_items = len(all_items)
        # Calculate the aggregate diversity
        aggregate_diversity = num_unique_items / total_items if total_items > 0 else 0.0
        # Return the aggregate diversity score
        return aggregate_diversity
