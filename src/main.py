import argparse
import src.knn as knn
import src.utils as ut
import src.recommenders as rec
import src.strategies as st
from src.data import Data
from src.utils import Recommendation, set_seed
from src.strategies import Strategy


def load_recommender(recommender_name: str, data: Data, k: int=5) -> rec.Recommender:
    """
    Load the specified recommender system. Available recommenders are:
    - popularity: Recommends the most popular items.
    - random: Recommends random items.
    - knn_user: User-based collaborative filtering recommender.
    - knn_item: Item-based collaborative filtering recommender.
    - UPDATE 1: Add more recommenders if needed.

    Args:
        recommender_name (str): Name of the recommender to load.
        data (Data): Data instance with the user-item interactions.
        k (int): Number of neighbors to consider for KNN recommenders.

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
    elif recommender_name == "knn_user":
        recommender = knn.UserBasedRatingPredictionRecommender(data, k=k)
    elif recommender_name == "knn_item":
        recommender = knn.ItemBasedRecommendationRecommender(data, k=k)
    else:
        print(f"Recommender {recommender_name} not found. Check the available "
              + "recommenders.")
        exit(1)

    return recommender

def load_strategy(strategy_name: str, data: Data) -> Strategy:
    """
    Load the specified recommendation strategy. Available strategies are:
    - exclude_seen: Exclude items already seen by the user.
    - no_filtering: Do not apply any filtering strategy.
    - UPDATE 2: Add more strategies if needed.

    Args:
        strategy_name (str): Name of the recommendation strategy to load.
        data (Data): Data instance with the user-item interactions.

    Returns:
        Strategy: Instance of the specified recommendation strategy.

    Raises:
        ValueError: If the strategy is not found.
    """
    # Load the strategy based on the specified name
    if strategy_name == "exclude_seen":
        strategy = st.ExcludeSeenStrategy(data)
    elif strategy_name == "no_filtering":
        strategy = st.NoFilteringStrategy(data)
    else:
        print(f"Strategy {strategy_name} not found. Check the available strategies.")
        exit(1)

    return strategy


def generate_recommendations(
    recommender_name: str,
    n_items_to_recommend: int,
    strategy_name: str,
    data: Data,
    k: int = 5,
) -> Recommendation:
    """
    Generate recommendations using the specified recommender system.

    Args:
        recommender_name (str): Name of the recommender to use.
        n_items_to_recommend (int): Number of recommendations to generate.
        strategy_name (str): Recommendation strategy to use.
        data (Data): Data instance with the user-item interactions.
        k (int): Number of neighbors to consider for KNN recommenders.

    Returns:
        Recommendation: Instance with the top-k recommendations for the user.
    """
    # Load the recommender
    recommender = load_recommender(recommender_name, data, k=k)

    # Load the strategy
    strategy = load_strategy(strategy_name, data)

    # Get the test users
    test_users = data.get_users(True)
    print(data.get_data()[1])

    # Generate recommendations
    for user in test_users:
        recommendations = recommender.recommend(user, strategy, n_items_to_recommend)
        # DUDA 4: Ver si lo guardo en un solo fichero o en varios
        # Save the recommendations
        # recommendations.save(f"data/recommendations_{user}.csv")
        # Print the recommendations
        print(f"Recommendations for user {user}:")
        print(recommendations)
        # TODO: Si si, saber que hay que juntar todos antes del return
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

    # FIXME 1: It has to be required
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

    # DUDA 2: Preguntar que era esto.
    # Recibir de entrenamiento y de test (quizas hacer otro main)
    # Es que si recibe dos archivos separados o uno y lo divide con el test_size?
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
        default=True,
    )

    # Add the column names argument
    parser.add_argument(
        "--col_names",
        type=list,
        help="column names for the data file",
        required=False,
        default=["user", "item", "rating", "timestamp"],
    )

    # Add the k argument for KNN recommenders
    parser.add_argument(
        "--k",
        type=int,
        help="number of neighbors to consider for KNN recommenders",
        required=False,
        default=5,
    )

    # Add the seed argument
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed for reproducibility",
        required=False,
        default=42,
    )

    # Parse the arguments
    args = parser.parse_args()

    # Set the random seed for reproducibility
    set_seed(args.seed)

    # Load the data
    data = Data(
        args.data_path, args.sep, args.test_size, args.ignore_first_line, args.col_names
    )

    # Generate recommendations
    recommendations = generate_recommendations(
        args.recommender,
        args.n_items_to_recommend,
        args.strategy,
        data,
        args.k
    )

    # TEST 5: Return the recommendations in a file user, item, score per row
    # Devolverlo en un fichero a parte usuario, item, score
    # Print the recommendations



if __name__ == "__main__":
    main()
