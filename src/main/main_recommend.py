import argparse
import os
import pandas as pd
import src.recommenders.basic_recommenders as rec
import src.recommenders.knn as knn
import src.recommenders.matrix_factorisation as mf
import src.recommenders.neuralnetworks as nn
import src.utils.similarities as sim
import src.utils.strategies as st
from src.datamodule.data import AbstractData, Data
from src.utils.utils import Recommendation, set_seed
from src.utils.similarities import Similarity
from src.utils.strategies import Strategy
from tqdm import tqdm


def load_recommender(
    recommender_name: str,
    data: AbstractData,
    k: int = 5,
    threshold: float = 1.0,
    similarity_measure: str = "pearson",
    n_factors: int = 20,
    lr: float = 0.01,
    reg: float = 0.1,
    n_epochs: int = 10,
    batch_size: int = 4096,
    device: str | None = None,
    hidden_dims: list[int] = [64, 32],
    n_layers: int = 2,
) -> rec.Recommender:
    """
    Load the specified recommender system. Available recommenders are:
    - popularity: Recommends the most popular items.
    - random: Recommends random items.
    - knn_user: User-based collaborative filtering recommender.
    - knn_item: Item-based collaborative filtering recommender.
    - mf: Matrix factorization recommender.
    - bprmf: Bayesian Personalized Ranking Matrix Factorization recommender.
    - mlp: Neural network recommender using a simple MLP on user and item embeddings.
    - gnn: Graph Neural Network recommender.
    - UPDATE 1: Add more recommenders if needed.

    Parameters:
        recommender_name (str): Name of the recommender to load.
        data (AbstractData): Data instance with the user-item interactions.
        k (int): Number of neighbors to consider for KNN recommenders.
        threshold (float): Threshold for KNN recommenders.
        similarity_measure (str | None): Similarity measure to use for KNN recommenders.
        n_factors (int): Number of latent factors for matrix factorization.
        lr (float): Learning rate for matrix factorization.
        reg (float): Regularization strength for matrix factorization.
        n_epochs (int): Number of epochs for matrix factorization.
        batch_size (int): Batch size for matrix factorization.
        device (str | None): Device to use for matrix factorization (e.g., "cuda", "cpu").
        hidden_dims (list[int]): Hidden dimensions for the MLP recommender.
        n_layers (int): Number of layers for the GNN recommender.

    Returns:
        Recommender: Instance of the specified recommender system.
    """
    # Load the recommender based on the specified name
    if recommender_name == "popularity":
        recommender = rec.PopularityRecommender(data)
    elif recommender_name == "random":
        recommender = rec.RandomRecommender(data)
    elif recommender_name == "knn_user":
        if similarity_measure:
            similarity_measure = load_similarity(similarity_measure, data, "user")
        recommender = knn.KNNUserBasedRecommender(
            data, k=k, similarity_measure=similarity_measure, threshold=threshold
        )
    elif recommender_name == "knn_item":
        if similarity_measure:
            similarity_measure = load_similarity(similarity_measure, data, "item")
        recommender = knn.KNNItemBasedRecommender(
            data, k=k, similarity_measure=similarity_measure
        )
    elif recommender_name == "mf":
        recommender = mf.MFRecommender(
            data,
            embedding_dim=n_factors,
            lr=lr,
            weight_decay=reg,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
        )
    elif recommender_name == "bprmf":
        recommender = mf.BPRMFRecommender(
            data,
            embedding_dim=n_factors,
            lr=lr,
            weight_decay=reg,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
        )
    elif recommender_name == "mlp":
        recommender = nn.MLPRecommender(
            data,
            embedding_dim=n_factors,
            hidden_dims=hidden_dims,
            lr=lr,
            weight_decay=reg,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
        )
    elif recommender_name == "gnn":
        recommender = nn.GNNRecommender(
            data,
            embedding_dim=n_factors,
            lr=lr,
            weight_decay=reg,
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_layers=n_layers,
            device=device,
        )
    else:
        print(f"Recommender {recommender_name} not found. Check the available "
              "recommenders.")
        exit(1)

    return recommender


def load_strategy(strategy_name: str, data: AbstractData) -> Strategy:
    """
    Load the specified recommendation strategy. Available strategies are:
    - exclude_seen: Exclude items already seen by the user.
    - no_filtering: Do not apply any filtering strategy.
    - UPDATE 2: Add more strategies if needed.

    Parameters:
        strategy_name (str): Name of the recommendation strategy to load.
        data (AbstractData): Data instance with the user-item interactions.

    Returns:
        Strategy: Instance of the specified recommendation strategy.
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


def load_similarity(similarity_name: str, data: AbstractData, mode: str) -> Similarity:
    """
    Load the specified similarity measure. Available measures are:
    - cosine: Cosine similarity.
    - pearson: Pearson correlation coefficient.
    - UPDATE 3: Add more similarity measures if needed.

    Parameters:
        similarity_name (str): Name of the similarity measure to load.
        data (AbstractData): Data instance with the user-item interactions.
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
              "similarity measures.")
        exit(1)

    return similarity


def generate_recommendations(
    recommender_name: str,
    n_items_to_recommend: int,
    strategy_name: str,
    data: AbstractData,
    k: int = 5,
    threshold: float = 1.0,
    similarity: str = "pearson",
    n_factors: int = 20,
    lr: float = 0.01,
    reg: float = 0.1,
    n_epochs: int = 10,
    batch_size: int = 4096,
    device: str | None = None,
    hidden_dims: list[int] = [64, 32],
    n_layers: int = 2,
) -> Recommendation:
    """
    Generate recommendations using the specified recommender system.

    Parameters:
        recommender_name (str): Name of the recommender to use.
        n_items_to_recommend (int): Number of recommendations to generate for each user.
        strategy_name (str): Name of the recommendation strategy to use.
        data (AbstractData): Data instance with the user-item interactions.
        k (int): Number of neighbors to consider for KNN recommenders.
        threshold (float): Threshold for KNN recommenders.
        similarity (str): Similarity measure to use for KNN recommenders.
        n_factors (int): Number of latent factors for matrix factorization.
        lr (float): Learning rate for matrix factorization.
        reg (float): Regularization strength for matrix factorization.
        n_epochs (int): Number of epochs for matrix factorization.
        batch_size (int): Batch size for matrix factorization.
        device (str | None): Device to use for matrix factorization (e.g., "cuda", "cpu").
        hidden_dims (list[int]): Hidden dimensions for the MLP recommender.
        n_layers (int): Number of layers for the GNN recommender.

    Returns:
        Recommendation: Instance with the top-k recommendations for the user.
    """
    # Load the recommender
    recommender = load_recommender(
        recommender_name,
        data,
        k=k,
        threshold=threshold,
        similarity_measure=similarity,
        n_factors=n_factors,
        lr=lr,
        reg=reg,
        n_epochs=n_epochs,
        batch_size=batch_size,
        device=device,
        hidden_dims=hidden_dims,
        n_layers=n_layers,
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
    for user in tqdm(test_users, desc="Generating recommendations"):
        recommendations = recommender.recommend(
            user, strategy, recommendations, n_items_to_recommend
        )

    return recommendations


def create_name(
    recommender_name: str,
    n_items_to_recommend: int,
    strategy_name: str,
    k: int | None = None,
    similarity_name: str | None = None,
    n_factors: int | None = None,
    lr: float | None = None,
    reg: float | None = None,
    n_epochs: int | None = None,
    batch_size: int | None = None,
    hidden_dims: list[int] | None = None,
    n_layers: int | None = None,
) -> str:
    """
    Create a name for the recommendation file based on the parameters used to generate
    the recommendations.

    Parameters:
        recommender_name (str): Name of the recommender used.
        n_items_to_recommend (int): Number of recommendations generated.
        strategy_name (str): Name of the recommendation strategy used.
        k (int): Number of neighbors considered for KNN recommenders.
        similarity_name (str): Similarity measure used for KNN recommenders.
        n_factors (int | None): Number of latent factors for matrix factorization.
        lr (float | None): Learning rate for matrix factorization.
        reg (float | None): Regularization strength for matrix factorization.
        n_epochs (int | None): Number of epochs for matrix factorization.
        batch_size (int | None): Batch size for matrix factorization.
        hidden_dims (list[int] | None): Hidden dimensions for the MLP recommender.
        n_layers (int | None): Number of layers for the GNN recommender.

    Returns:
        str: Name of the recommendation file.
    """
    parts = [recommender_name, str(n_items_to_recommend), strategy_name]

    # Detalle extra para KNN
    if recommender_name in {"knn_user", "knn_item"}:
        parts.extend([f"k{k}", similarity_name])

    # Detalle extra para Matrix Factorization
    elif recommender_name == "mf":
        parts.extend([
            f"f{n_factors}",
            f"lr{str(lr).replace('.', 'p')}",
            f"reg{str(reg).replace('.', 'p')}",
            f"ep{n_epochs}",
            f"bs{batch_size}",
        ])

    # Detalle extra para BPR-MF
    elif recommender_name == "bprmf":
        parts.extend([
            f"f{n_factors}",
            f"lr{str(lr).replace('.', 'p')}",
            f"reg{str(reg).replace('.', 'p')}",
            f"ep{n_epochs}",
            f"bs{batch_size}",
        ])

    elif recommender_name == "mlp":
        parts.extend([
            f"f{n_factors}",
            f"lr{str(lr).replace('.', 'p')}",
            f"hd{'-'.join(map(str, hidden_dims))}",
            f"reg{str(reg).replace('.', 'p')}",
            f"ep{n_epochs}",
            f"bs{batch_size}",
        ])

    elif recommender_name == "gnn":
        parts.extend([
            f"f{n_factors}",
            f"lr{str(lr).replace('.', 'p')}",
            f"reg{str(reg).replace('.', 'p')}",
            f"ep{n_epochs}",
            f"bs{batch_size}",
            f"nl{n_layers}",
        ])

    filename = "_".join(parts) + ".csv"
    return filename


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
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Mini-batch size for matrix factorization",
        required=False,
        default=4096,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="cuda, cpu or leave empty for auto-detect",
        required=False,
        default=None,
    )
    # MLP Arguments
    # Add the hidden dimensions argument for the MLP recommender
    parser.add_argument(
        "--hidden_dims",
        nargs="+",
        type=int,
        help="Hidden layer sizes for the MLP recommender, e.g. --hidden_dims 64 32",
        required=False,
        default=[64, 32],
    )

    # GNN Arguments
    # Add the number of layers argument for the GNN recommender
    parser.add_argument(
        "--n_layers",
        type=int,
        help="Number of layers for the GNN recommender",
        required=False,
        default=2,
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
        default="results/recommendations/",
    )
    # Parse the arguments
    args = parser.parse_args()

    # Set the random seed for reproducibility
    set_seed(args.seed)

    # Load the data
    try:
        data = Data(
            args.data_path_train,
            args.data_path_test,
            args.sep,
            args.test_size,
            args.ignore_first_line,
            args.col_names,
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

    # Generate recommendations
    recommendations = generate_recommendations(
        recommender_name=args.recommender,
        n_items_to_recommend=args.n_items_to_recommend,
        strategy_name=args.strategy,
        data=data,
        k=args.k,
        threshold=args.threshold,
        similarity=args.similarity,
        n_factors=args.n_factors,
        lr=args.lr,
        reg=args.reg,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=args.device,
        hidden_dims=args.hidden_dims,
        n_layers=args.n_layers,
    )

    # Create the name for the recommendation file
    name = create_name(
        recommender_name=args.recommender,
        n_items_to_recommend=args.n_items_to_recommend,
        strategy_name=args.strategy,
        k=args.k,
        similarity_name=args.similarity,
        n_factors=args.n_factors,
        lr=args.lr,
        reg=args.reg,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        hidden_dims=args.hidden_dims,
        n_layers=args.n_layers,
    )

    # If the save path does not end with a slash, add it
    if not args.save_path.endswith("/"):
        args.save_path += "/"
    # If the save path does not exist, create it
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Include the path to the file
    name = args.save_path + name
    # Save the recommendations
    recommendations.save(name, mode="w")
    print(f"Recommendations saved to {name}")


if __name__ == "__main__":
    main()
