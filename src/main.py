import argparse
import src.utils as ut
import src.recommenders as rec
import src.strategies as st
from src.data import Data
from src.utils import Recommendation
from src.strategies import Strategy


def load_recommender(recommender_name: str, data: Data) -> rec.Recommender:
    """
    Load the specified recommender system. Available recommenders are:
    - popularity: Recommends the most popular items.
    - random: Recommends random items.
    - XXX: Add more recommenders if needed.

    Args:
        recommender_name (str): Name of the recommender to load.

    Returns:
        Recommender: Instance of the specified recommender system.

    Raises:
        ValueError: If the recommender is not found.
    """
    # Load the recommender based on the specified name
    if recommender_name == "popularity":
        recommender = rec.PopularityRecommender(data)
    elif recommender_name == "random":
        recommender = rec.RandomRecommender(data)
    else:
        print(f"Recommender {recommender_name} not found. Check the available "
              + "recommenders.")
        exit(1)

    return recommender


def load_strategy(strategy_name: str) -> Strategy:
    """
    Load the specified recommendation strategy. Available strategies are:
    - exclude_seen: Exclude items already seen by the user.
    - no_filtering: Do not apply any filtering strategy.
    - XXX: Add more strategies if needed.

    Args:
        strategy_name (str): Name of the recommendation strategy to load.

    Returns:
        Strategy: Instance of the specified recommendation strategy.

    Raises:
        ValueError: If the strategy is not found.
    """
    # Load the strategy based on the specified name
    if strategy_name == "exclude_seen":
        strategy = st.ExcludeSeenStrategy()
    elif strategy_name == "no_filtering":
        strategy = st.NoFilteringStrategy()
    else:
        print(f"Strategy {strategy_name} not found. Check the available strategies.")
        exit(1)

    return strategy


def generate_recommendations(
    recommender_name: str,
    user: int,
    n_items_to_recommend: int,
    strategy_name: str,
    data: Data,
) -> Recommendation:
    """
    Generate recommendations using the specified recommender system.

    Args:
        recommender_name (str): Name of the recommender to use.
        user (int): User ID to generate recommendations for.
        n_items_to_recommend (int): Number of recommendations to generate.
        strategy_name (str): Recommendation strategy to use.
        data (Data): Data instance with the user-item interactions.

    Returns:
        Recommendation: Instance with the top-k recommendations for the user.
    """
    # Load the recommender
    recommender = load_recommender(recommender_name, data)

    # Load the strategy
    strategy = load_strategy(strategy_name)

    # Generate recommendations
    recommendations = recommender.recommend(user, strategy, n_items_to_recommend)
    return recommendations


def main():
    """
    Main function for the recommender system script.
    """
    parser = argparse.ArgumentParser(
        description="Generate recommendations using specified recommender system."
    )
    # Add the recommender argument
    parser.add_argument(
        "--recommender", type=str, help="name of the recommender to use", required=True
    )
    # Add the number of recommendations argument
    parser.add_argument(
        "--n_items_to_recommend",
        type=int,
        help="number of recommendations to generate",
        required=False,
        default=10,
    )
    # Add the recommendation strategy argument
    parser.add_argument(
        "--strategy",
        type=str,
        help="recommendation strategy to use",
        required=False,
        default="exclude_seen",
    )

    # FIXME: It has to be required
    # Add the data path argument
    parser.add_argument(
        "--data_path",
        type=str,
        help="path to the data file",
        required=False,
        default="data/ml-100k/u1.base",
    )

    # Add the data separator argument
    parser.add_argument(
        "--sep",
        type=str,
        help="separator for the data file",
        required=False,
        default="\t",
    )

    # TODO: Recibir de entrenamiento y de test (quizas hacer otro main)
    # Add the test size argument
    parser.add_argument(
        "--test_size",
        type=float,
        help="proportion of data to split into test set",
        required=False,
        default=0.2,
    )

    # Add the ignore first line argument
    parser.add_argument(
        "--ignore_first_line",
        type=bool,
        help="whether to ignore the first line of the data file",
        required=False,
        default=False,
    )

    # Add the column names argument
    parser.add_argument(
        "--col_names",
        type=list,
        help="column names for the data file",
        required=False,
        default=["user", "item", "rating", "timestamp"],
    )

    # Parse the arguments
    args = parser.parse_args()

    # Load the data
    data = Data(
        args.data_path, args.sep, args.test_size, args.ignore_first_line, args.col_names
    )

    # Generate recommendations
    recommendations = generate_recommendations(
        args.recommender,
        args.user,
        args.n_items_to_recommend,
        args.strategy,
        data,
    )

    # TODO: Return the recommendations in a file user, item, score per row
    recommendations.save("data/recommendations.csv")
    # Devolverlo en un fichero a parte usuario, item, score
    # Print the recommendations
    print(recommendations)


if __name__ == "__main__":
    main()
