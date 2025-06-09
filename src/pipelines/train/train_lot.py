import itertools
import subprocess

"""
This script runs a grid search for different recommender models.
Complete the required parameters in the main section and run the script to execute the
grid search for the specified model.
"""


# Fixed arguments
BASE_CMD = [
    "python", "-m", "src.main.main_recommend",
    "--n_items_to_recommend", "5",
    "--save_path", "results/recommendations/NewYork_10",
]


def run_knn_search(model: str):
    """
    Run grid search for KNN-based models.

    Args:
        model (str): The name of the model to run ("knn_user" or "knn_item").
    """
    if model not in {"knn_user", "knn_item"}:
        raise ValueError("Model must be 'knn_user' or 'knn_item'")

    PARAM_GRID = {
        "--k": [1, 5, 10],
        "--similarity": ["cosine", "pearson"],
    }

    print(f"\nStarting grid search for model: {model.upper()}")
    total_runs = len(list(itertools.product(*PARAM_GRID.values())))

    for i, combo in enumerate(itertools.product(*PARAM_GRID.values()), 1):
        cmd = BASE_CMD.copy()
        cmd.extend(["--recommender", model])

        for key, value in zip(PARAM_GRID.keys(), combo):
            cmd.extend([key, str(value)])

        print(f"\nRunning combination {i}/{total_runs}:")
        print(" ".join(cmd))
        subprocess.run(cmd)


def run_grid_search(model: str):
    """
    Run grid search with the specified model type ('mf', 'bprmf', 'mlp', or 'gnn').

    Args:
        model (str): The name of the model to run ("mf" or "bprmf" or "mlp" or "gnn").
    """
    # Hyperparameter grid for both models
    param_grid = {
        # "--n_factors": [32, 64, 128],
        # "--lr": [0.01, 0.005],
        # "--reg": [0.1, 0.01],
        # "--n_epochs": [50, 100],
        # "--batch_size": [2048, 4096]
        "--n_factors": [256, 512],
        "--lr": [0.01, 0.005],
        "--reg": [0.1, 0.01],
        "--n_epochs": [100],
        "--batch_size": [2048]
    }

    if model == "mlp":
        param_grid["--hidden_dims"] = [(128, 64), (256, 128)]

    if model == "gnn":
        param_grid["--n_layers"] = [2, 3]

    print(f"\nStarting grid search for model: {model.upper()}")
    total_runs = len(list(itertools.product(*param_grid.values())))

    for i, combo in enumerate(itertools.product(*param_grid.values()), 1):
        cmd = BASE_CMD.copy()
        cmd.extend(["--recommender", model])

        for key, value in zip(param_grid.keys(), combo):
            if key == "--hidden_dims":
                cmd.extend(["--hidden_dims"] + [str(x) for x in value])
            else:
                cmd.extend([key, str(value)])

        print(f"\nRunning combination {i}/{total_runs}:")
        print(" ".join(cmd))
        subprocess.run(cmd)


if __name__ == "__main__":
    ########## CONFIGURATION ##########
    # Choose a model between:
    # "popularity" "random" "knn_user" "knn_item" "mf" "bprmf" "mlp" "gnn"
    model = "mlp"

    # Choose a strategy between:
    # "exclude_seen" "no_filtering"
    strategy = "no_filtering"

    # Info for the data
    data_path = "data/dataset/train.txt"
    sep = "\t"
    test_file = True

    # For my use:
    data_path = "data/ml-100k/u1.base"  # Path to the training data file
    test_file = True
    data_path = "data/NewYork/US_NewYork_Processed_Shortened_10.txt"
    test_file = False

    if test_file:
        test_path = "data/dataset/test.txt"
        # For my use:
        test_path = "data/ml-100k/u1.base"
        BASE_CMD.append("--data_path_test")
        BASE_CMD.append(test_path)
    else:
        test_size = 0.1
        BASE_CMD.append("--test_size")
        BASE_CMD.append(str(test_size))

    # Include the parameters in the command
    BASE_CMD.append("--data_path_train")
    BASE_CMD.append(data_path)
    BASE_CMD.append("--sep")
    BASE_CMD.append(sep)

    if model == "bprmf" or model == "mf" or model == "mlp" or model == "gnn":
        run_grid_search(model)
    elif model == "knn_user" or model == "knn_item":
        run_knn_search(model)
    elif model == "popularity" or model == "random":
        # For popularity and random, no grid search is needed
        BASE_CMD.append("--recommender")
        BASE_CMD.append(model)
        print(f"\nRunning {model.upper()} recommender:")
        print(" ".join(BASE_CMD))
        subprocess.run(BASE_CMD)
    else:
        print(f"Unknown model: {model}. Please choose from 'popularity', 'random',"
              "'knn_user', 'knn_item', 'mf', 'bprmf', 'mpl', or 'gnn'.")
