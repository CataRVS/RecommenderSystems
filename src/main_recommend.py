import argparse
import src.knn as knn
import src.matrix_factorisation as mf
import src.recommenders as rec
import src.similarities as sim
import src.strategies as st
from src.data import Data
from src.utils import Recommendation, set_seed
from src.similarities import Similarity
from src.strategies import Strategy


def load_recommender(
    recommender_name: str,
    data: Data,
    k: int = 5,
    threshold: float = 1.0,
    similarity_measure: str | None = None,
    n_factors: int = 20,
    lr: float = 0.01,
    reg: float = 0.1,
    n_epochs: int = 10
) -> rec.Recommender:
    """
    Load the specified recommender system. Available recommenders are:
    - popularity: Recommends the most popular items.
    - random: Recommends random items.
    - knn_user: User-based collaborative filtering recommender.
    - knn_item: Item-based collaborative filtering recommender.
    - mf: Matrix factorization recommender.
    - UPDATE 1: Add more recommenders if needed.

    Parameters:
        recommender_name (str): Name of the recommender to load.
        data (Data): Data instance with the user-item interactions.
        k (int): Number of neighbors to consider for KNN recommenders.
        threshold (float): Threshold for KNN recommenders.
        similarity_measure (Similarity | None): Similarity measure to use for KNN recommenders.
        n_factors (int): Number of latent factors for matrix factorization.
        lr (float): Learning rate for matrix factorization.
        reg (float): Regularization strength for matrix factorization.
        n_epochs (int): Number of epochs for matrix factorization.

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
        if similarity_measure:
            similarity_measure = load_similarity(similarity_measure, data, "user")
        recommender = knn.UserBasedRankingRecommender(
            data, k=k, similarity_measure=similarity_measure, threshold=threshold
        )
    elif recommender_name == "knn_item":
        if similarity_measure:
            similarity_measure = load_similarity(similarity_measure, data, "item")
        recommender = knn.ItemBasedRecommendationRecommender(
            data, k=k, similarity_measure=similarity_measure
        )
    elif recommender_name == "mf":
        recommender = mf.MFRecommender(
            data,
            n_factors=n_factors,
            lr=lr,
            regularization=reg,
            n_epochs=n_epochs
        )
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

    Parameters:
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


def load_similarity(
    similarity_name: str, data: Data, mode: str
) -> Similarity:
    """
    Load the specified similarity measure. Available measures are:
    - cosine: Cosine similarity.
    - pearson: Pearson correlation coefficient.
    - UPDATE 3: Add more similarity measures if needed.

    Parameters:
        similarity_name (str): Name of the similarity measure to load.
        data (Data): Data instance with the user-item interactions.
        mode (str): Mode of the recommender system (user or item).

    Returns:
        Similarity: Instance of the specified similarity measure.
    """
    # Load the similarity measure based on the specified name
    if similarity_name == "cosine":
        if mode == "user":
            similarity = sim.CosineSimilarityUsers(data)
        else:
            similarity = sim.CosineSimilarityItems(data)
    elif similarity_name == "pearson":
        if mode == "user":
            similarity = sim.PearsonCorrelationUsers(data)
        else:
            similarity = sim.PearsonCorrelationItems(data)
    else:
        print(f"Similarity measure {similarity_name} not found. Check the available "
              + "similarity measures.")
        exit(1)

    return similarity


def generate_recommendations(
    recommender_name: str,
    n_items_to_recommend: int,
    strategy_name: str,
    data: Data,
    k: int = 5,
    threshold: float = 1.0,
    similarity: str = "pearson",
) -> Recommendation:
    """
    Generate recommendations using the specified recommender system.

    Parameters:
        recommender_name (str): Name of the recommender to use.
        n_items_to_recommend (int): Number of recommendations to generate.
        strategy_name (str): Recommendation strategy to use.
        data (Data): Data instance with the user-item interactions.
        k (int): Number of neighbors to consider for KNN recommenders.

    Returns:
        Recommendation: Instance with the top-k recommendations for the user.
    """
    # Load the recommender
    recommender = load_recommender(
        recommender_name, data, k=k, threshold=threshold, similarity_measure=similarity
    )

    # Load the strategy
    strategy = load_strategy(strategy_name, data)

    # Get the internal test‐set IDs, then map them back to the original IDs
    internal_users = data.get_users(test=True)
    uidx_map, _ = data.get_reverse_mappings()
    # If a user is not in the training set, it will be excluded from the recommendations
    # Map the internal user IDs to the original user IDs
    test_users = [uidx_map.get(user) for user in internal_users]
    # Filter out users that are not in the training set
    test_users = [uid for uid in test_users if uid is not None]

    # Instance the recommendation object
    recommendations = Recommendation()

    # Generate recommendations
    for user in test_users:
        recommendations = recommender.recommend(user, strategy, recommendations, n_items_to_recommend)

    return recommendations


def create_name(
    recommender_name: str,
    n_items_to_recommend: int,
    strategy_name: str,
    k: int = 5,
    similarity_name: str = "pearson",
) -> str:
    """
    Create a name for the recommendation file based on the parameters used to generate
    the recommendations.

    Returns:
        str: Name of the recommendation file.
    """
    name = f"{recommender_name}_{n_items_to_recommend}_{strategy_name}"
    if recommender_name == "knn_user" or recommender_name == "knn_item":
        name += f"_k{k}_{similarity_name}"

    name += ".csv"

    return name


def main():
    """
    Main function for the recommender system script.
    """
    parser = argparse.ArgumentParser(
        description="Generate recommendations using specified recommender system."
    )
    # Add the recommender argument
    parser.add_argument(
        "--recommender",
        type=str,
        help="Name of the recommender to use",
        required=True
    )
    # Add the number of recommendations argument
    parser.add_argument(
        "--n_items_to_recommend",
        type=int,
        help="Number of recommendations to generate",
        required=False,
        default=10,
    )
    # Add the recommendation strategy argument
    parser.add_argument(
        "--strategy",
        type=str,
        help="Recommendation strategy to use",
        required=False,
        default="exclude_seen",
    )
    # Add the train data path argument
    parser.add_argument(
        "--data_path_train",
        type=str,
        help="Path to the training data file",
        required=True
    )
    # Add the test data path argument
    parser.add_argument(
        "--data_path_test",
        type=str,
        help="Path to the test data file",
        required=False,
        default="none",
    )
    # Add the test size argument
    parser.add_argument(
        "--test_size",
        type=float,
        help="Proportion of data to split into test set",
        required=False,
        default=0.2,
    )
    # Add the data separator argument
    parser.add_argument(
        "--sep",
        type=str,
        help="Separator used in the data files",
        required=False,
        default="\t",
    )
    # Add the ignore first line argument
    parser.add_argument(
        "--ignore_first_line",
        type=bool,
        help="Whether to ignore the first line of the data files",
        required=False,
        default=False,
    )
    # Add the column names argument
    parser.add_argument(
        "--col_names",
        type=list,
        help="Column names for the data file",
        required=False,
        default=["user", "item", "rating", "timestamp"],
    )

    # KNN Arguments
    # Add the k argument for KNN recommenders
    parser.add_argument(
        "--k",
        type=int,
        help="Number of neighbors to consider for KNN recommenders",
        required=False,
        default=5,
    )
    # Add the threshold argument for KNN recommenders
    parser.add_argument(
        "--threshold",
        type=float,
        help="Threshold for KNN recommenders",
        required=False,
        default=1.0,
    )
    # Add the similarity measure argument for KNN recommenders
    parser.add_argument(
        "--similarity",
        type=str,
        help="Similarity measure to use for KNN recommenders",
        required=False,
        default="pearson",
    )

    # MATRIX FACTORIZATION Arguments
    # Add the number of factors argument for matrix factorization
    parser.add_argument(
        "--n_factors",
        type=int,
        help="Number of latent factors for matrix factorization",
        required=False,
        default=20,
    )
    # Add the learning rate argument for matrix factorization
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate for matrix factorization",
        required=False,
        default=0.01,
    )
    # Add the regularization argument for matrix factorization
    parser.add_argument(
        "--reg",
        type=float,
        help="Regularization strength for matrix factorization",
        required=False,
        default=0.1,
    )
    # Add the number of epochs argument for matrix factorization
    parser.add_argument(
        "--n_epochs",
        type=int,
        help="Number of epochs for matrix factorization",
        required=False,
        default=10,
    )

    # Add the seed argument
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
        required=False,
        default=42,
    )
    # Add the save path argument
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to save the recommendations",
        required=False,
        default="recommendations/",
    )
    # Parse the arguments
    args = parser.parse_args()

    # Set the random seed for reproducibility
    set_seed(args.seed)

    # Load the data
    data = Data(
        args.data_path_train,
        args.data_path_test,
        args.sep,
        args.test_size,
        args.ignore_first_line,
        args.col_names,
    )

    # Generate recommendations
    recommendations = generate_recommendations(
        args.recommender,
        args.n_items_to_recommend,
        args.strategy,
        data,
        args.k,
        args.threshold,
        similarity=args.similarity,
    )

    # Create the name for the recommendation file
    name = create_name(
        args.recommender,
        args.n_items_to_recommend,
        args.strategy,
        args.k,
        args.similarity,
    )
    # Include the path to the file
    name = args.save_path + name
    # Save the recommendations
    recommendations.save(name, mode="w")
    print(f"Recommendations saved to {name}")


if __name__ == "__main__":
    main()
