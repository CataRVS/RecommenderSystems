import argparse
import pandas as pd
import src.evaluation.evaluation as ev
import src.datamodule.splits as sp
from src.datamodule.data import AbstractData, Data
from src.utils.utils import set_seed


def load_data(
    data_path_train: str,
    data_path_test: str | None = None,
    sep: str = "\t",
    test_size: float = 0.2,
    ignore_first_line: bool = False,
    split_strategy: str = "random",
    seed: int = 42,
) -> AbstractData:
    """
    Load the data from the specified paths and split it following one of the following strategies:
    - random: Randomly split the data into training and test sets.
    - leave_one_last: Leave the last interaction of each user as the test set.
    - temporal_user: Split the data temporally for each user.
    - temporal_global: Split the data temporally for the entire dataset.
    - UPDATE 1: Add more split strategies if needed.

    Parameters:
        data_path_train (str): Path to the training data file.
        data_path_test (str | None): Path to the test data file. If None, a test set
                                      will be created from the training data.
        sep (str): Separator used in the data files.
        test_size (float): Proportion of the dataset to include in the test split.
        ignore_first_line (bool): Whether to ignore the first line of the data files.
        split_strategy (str): Strategy to split the data into train and test sets.
        seed (int): Random seed for reproducibility.

    Returns:
        AbstractData: An instance of AbstractData containing the loaded data.
    """
    # Load the data using the Data class
    data = Data(
        data_path_train,
        data_path_test,
        sep,
        test_size,
        ignore_first_line,
    )

    if data_path_test == "none" and split_strategy == "none":
        split_strategy = "random"

    if split_strategy == "random":
        splitter = sp.RandomSplitter(test_size=test_size, seed=seed)
    elif split_strategy == "leave_one_last":
        splitter = sp.LeaveOneLastSplitter()
    elif split_strategy == "temporal_user":
        splitter = sp.TemporalUserSplitter(test_size=test_size)
    elif split_strategy == "temporal_global":
        splitter = sp.TemporalGlobalSplitter(test_size=test_size)
    elif split_strategy == "none":
        return data
    else:
        print(f"Split strategy {split_strategy} not found. "
              "Check the available split strategies.")
        exit(1)

    # Split the data using the specified strategy
    data = splitter.split(data)

    return data


def load_evaluation_metric(evaluation_name: str, data: AbstractData) -> ev.Evaluation:
    """
    Load the evaluation metric based on the provided name. Available metrics are:
    - precision
    - recall
    - ndcg
    - epc
    - gini
    - aggregate_diversity
    - UPDATE 5: Add more metrics if needed.

    Parameters:
        evaluation_name (str): Name of the evaluation metric.
        data (AbstractData): Data object containing user-item interactions.

    Returns:
        Evaluation: Instance of the evaluation metric class.
    """
    # Load the evaluation metric based on the provided name
    if evaluation_name == "precision":
        evaluator = ev.Precision(data)
    elif evaluation_name == "recall":
        evaluator = ev.Recall(data)
    elif evaluation_name == "ndcg":
        evaluator = ev.NDCG(data)
    elif evaluation_name == "epc":
        evaluator = ev.EPC(data)
    elif evaluation_name == "gini":
        evaluator = ev.Gini(data)
    elif evaluation_name == "aggregate_diversity":
        evaluator = ev.AggregateDiversity(data)
    else:
        print(f"Evaluation metric '{evaluation_name}' not found. Check the available "
              "metrics.")
        exit(1)

    return evaluator


def evaluate_recommendations(
    evaluation_name: str,
    data_path_recommendations: str,
    recommendations_sep: str,
    ignore_first_line_recs: bool,
    data: AbstractData
):
    """
    Evaluate the recommendations using the specified evaluation metric.

    Parameters:
        evaluation_name (str): Name of the evaluation metric to use.
        data_path_recommendations (str): Path to the recommendations file.
        recommendations_sep (str): Separator used in the recommendations file.
        ignore_first_line_recs (bool): Whether to ignore the first line of the recommendations file.
        data (AbstractData): Data object containing user-item interactions.
    """
    # Load the evaluation metric
    evaluator = load_evaluation_metric(evaluation_name, data)

    # Evaluate the recommendations
    score = evaluator.evaluate(
        recommendations_path=data_path_recommendations,
        recommendations_sep=recommendations_sep,
        ignore_first_line=ignore_first_line_recs,
    )

    # Print the score
    print(f"{evaluation_name.capitalize()} score: {score:.4f}")


def main():
    """
    Main function for the evaluation script. Parses command line arguments and
    evaluates the recommendations.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the recommendations using different metrics."
    )
    # Add the metric name argument
    parser.add_argument(
        "--metric",
        type=str,
        help="Name of the evaluation metric to use.",
        required=True,
    )
    # Add the recommendations path argument
    parser.add_argument(
        "--recommendations_path",
        type=str,
        help="Path to the recommendations file.",
        required=True,
    )
    # Add train data path argument
    parser.add_argument(
        "--data_path_train",
        type=str,
        help="Path to the training data file.",
        required=True,
    )
    # Add test data path argument
    parser.add_argument(
        "--data_path_test",
        type=str,
        help="Path to the testing data file.",
        required=False,
        default="none",
    )
    # Add the test size argument
    parser.add_argument(
        "--test_size",
        type=float,
        help="Proportion of the data to include in the test split.",
        required=False,
        default=0.2,
    )
    # Add the recomendations separator argument
    parser.add_argument(
        "--sep_recs",
        type=str,
        help="Separator used in the recommendations file.",
        required=False,
        default=",",
    )
    # Add the data separator argument
    parser.add_argument(
        "--sep_data",
        type=str,
        help="Separator used in the data files.",
        required=False,
        default="\t",
    )
    # Add the ignore first line argument in the recommendations file
    parser.add_argument(
        "--ignore_first_line_recs",
        type=bool,
        help="Whether to ignore the first line of the recommendations file.",
        required=False,
        default=False,
    )
    # Add the ignore first line argument in the data files
    parser.add_argument(
        "--ignore_first_line_data",
        type=bool,
        help="Whether to ignore the first line of the data files.",
        required=False,
        default=False,
    )
    # Add the split strategy argument
    parser.add_argument(
        "--split_strategy",
        type=str,
        help="Strategy to split the data into train and test sets. "
             "Available strategies: random, leave_one_last, temporal_user, temporal_global, none",
        required=False,
        default="none",
    )
    # Add the random seed argument
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility.",
        required=False,
        default=42,
    )
    # Parse the arguments
    args = parser.parse_args()

    # Set the random seed
    set_seed(args.seed)

    # Load the data
    # Load the data
    try:
        data = load_data(
            data_path_train=args.data_path_train,
            data_path_test=args.data_path_test,
            sep=args.sep_data,
            test_size=args.test_size,
            ignore_first_line=args.ignore_first_line_data,
            split_strategy=args.split_strategy,
            seed=args.seed,
        )
    except FileNotFoundError as e:
        print(e)
        exit(1)
    except pd.errors.ParserError as e:
        print(e)
        exit(1)
    except ValueError as e:
        print(e)
        exit(1)

    # Evaluate the recommendations
    evaluate_recommendations(
        evaluation_name=args.metric,
        data_path_recommendations=args.recommendations_path,
        recommendations_sep=args.sep_recs,
        ignore_first_line_recs=args.ignore_first_line_recs,
        data=data,
    )


if __name__ == "__main__":
    main()
