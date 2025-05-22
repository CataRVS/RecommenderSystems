from abc import ABC, abstractmethod
from src.data import Data
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
            # Calculate recall
            recalls.append(tp / len(ground_truth))

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
    Computes the expected precision (EP) of the recommendations.
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
        recommendations = self.data._load_recs(
            recommendations_path, recommendations_sep, ignore_first_line
        )

        # Calculate expected precision for each user
        ep_scores = []
        for u, items in recommendations.items():
            # ground‐truth items in the test set
            gt = set(self.data.get_test_interactions(u).keys())
            if not items:
                continue

            # build binary relevance vector
            rel = [1 if i in gt else 0 for i in items]

            # compute precision@j for each prefix j
            cum_rel = 0
            precisions = []
            for j, r in enumerate(rel, start=1):
                cum_rel += r
                precisions.append(cum_rel / j)

            # EP(u) = mean_j precision@j
            ep_scores.append(sum(precisions) / len(precisions))

        # return macro‐averaged EP
        return float(np.mean(ep_scores)) if ep_scores else 0.0


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
        # We count the occurrences of each item
        item_counts = np.bincount(items)
        # We sort the counts to compute the Gini coefficient
        sorted_counts = np.sort(item_counts)
        # If there are no items, return 0.0
        n = len(sorted_counts)
        if n == 0:
            return 0.0
        # Calculate the Gini coefficient
        # Gini = (1 / (n-1)) * Σ(j=1 to n) (2j - n - 1) * p(i_j)
        # Gini = Σ(i=1 to n) Σ(j=1 to n) |x_i - x_j| / (2 * n^2 * μ)
        # where μ is the mean of the counts

        # Σ(i=1 to n) Σ(j=1 to n) |x_i - x_j|
        numerator = np.sum(np.abs(sorted_counts[:, None] - sorted_counts[None, :]))
        # 2 * n^2 * μ
        denominator = 2 * n**2 * np.mean(sorted_counts)

        # Gini coefficient
        gini = numerator / denominator

        # Return the Gini coefficient
        return gini


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
