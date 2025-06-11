import argparse
import pandas as pd
import src.evaluation.evaluation as ev
from src.datamodule.data import AbstractData, Data
from src.utils.utils import set_seed


def load_evaluation_metric(evaluation_name: str, data: AbstractData) -> ev.Evaluation:
    """
    Load the evaluation metric based on the provided name. Available metrics are:
    - precision
    - recall
    - ndcg
    - epc
    - gini
    - aggregate_diversity
    - UPDATE 4: Add more metrics if needed.

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
    try:
        data = Data(
            data_path_train=args.data_path_train,
            data_path_test=args.data_path_test,
            sep=args.sep_data,
            test_size=args.test_size,
            ignore_first_line=args.ignore_first_line_data,
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
