import os
import argparse
import pandas as pd
from tqdm import tqdm

from src.datamodule.data import Data
from src.utils.utils import set_seed
import src.evaluation.evaluation as ev


# Constants
METRICS = ["precision", "recall", "ndcg", "epc", "gini", "aggregate_diversity"]
RECS_SEP = ","
DEFAULT_COLS = ["user", "item", "rating", "timestamp"]


def load_data(
    train_path: str,
    test_path: str = "none",
    sep: str = "\t",
    test_size: float = 0.2,
    ignore_first: bool = False,
    seed: int = 42
) -> Data:
    """
    Load the dataset for evaluation.

    Parameters:
        train_path (str): Path to the training data file.
        test_path (str): Path to the test data file. If "none", no test data is loaded.
        sep (str): Separator used in the data files.
        test_size (float): Proportion of the dataset to include in the test split.
        ignore_first (bool): Whether to ignore the first line of the data files.
        seed (int): Random seed for reproducibility.

    Returns:
        Data: An instance of the Data class containing the loaded dataset.
    """
    set_seed(seed)
    data = Data(
        data_path_train=train_path,
        data_path_test=test_path if test_path.lower() != "none" else "none",
        sep=sep,
        test_size=test_size,
        ignore_first_line=ignore_first,
        col_names=DEFAULT_COLS,
    )
    return data


def load_evaluation_metric(metric_name: str, data: Data) -> ev.Evaluation:
    match metric_name:
        case "precision": return ev.Precision(data)
        case "recall": return ev.Recall(data)
        case "ndcg": return ev.NDCG(data)
        case "epc": return ev.EPC(data)
        case "gini": return ev.Gini(data)
        case "aggregate_diversity": return ev.AggregateDiversity(data)
        case _: raise ValueError(f"Metric {metric_name} is not supported.")


def parse_filename(filename):
    """
    Parse the filename to extract recommender, strategy, parameters, and configuration.

    Parameters:
        filename (str): Name of the recommendation CSV file.

    Returns:
        tuple: (recommender, strategy, params, config)
            - recommender (str): Name of the recommender.
            - strategy (str): Strategy used by the recommender.
            - params (dict): Dictionary of parameters extracted from the filename.
            - config (str): String representation of the configuration.
    """
    base = os.path.basename(filename).replace(".csv", "")
    parts = base.split("_")

    if filename.startswith("knn_"):
        recommender = parts[0] + "_" + parts[1]
        strategy = parts[3] + "_" + parts[4]
        k = int(parts[5][1:])
        similarity = parts[6]
        params = {"k": k, "similarity": similarity}
        config = f"k{k}_{similarity}"
    elif filename.startswith("mf_") or filename.startswith("bprmf_"):
        recommender = parts[0]
        strategy = parts[2] + "_" + parts[3]
        n_factors = int(parts[4][1:])
        learning_rate = float(parts[5][2:].replace("p", "."))
        regularization = float(parts[6][3:].replace("p", "."))
        epochs = int(parts[7][2:])
        batch_size = int(parts[8][2:])
        params = {
            "n_factors": n_factors,
            "learning_rate": learning_rate,
            "regularization": regularization,
            "epochs": epochs,
            "batch_size": batch_size
        }
        config = f"f{n_factors}_lr{learning_rate}_reg{regularization}_ep{epochs}_bs{batch_size}"
    else:
        recommender = parts[0]
        strategy = parts[2] + "_" + parts[3]
        params = {}
        config = ""

    return recommender, strategy, params, config


def evaluate_file(file_path, data):
    scores = {}
    for metric in METRICS:
        evaluator = load_evaluation_metric(metric, data)
        score = evaluator.evaluate(
            recommendations_path=file_path,
            recommendations_sep=RECS_SEP,
            ignore_first_line=False,
        )
        scores[metric] = score
    return scores


def export_results_to_csv(results, save_path):
    rows = []
    for file, (recommender, strategy, params, config, scores) in results:
        row = {
            "filename": file,
            "recommender": recommender,
            "strategy": strategy,
            "config": config,
        }
        row.update(params)
        row.update(scores)
        rows.append(row)

    path = os.path.join(save_path, "bprmf_NY_10_evaluation_results.csv")
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recs_dir", required=True, help="Folder containing recommendation CSV files")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", default="none")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--data_sep", default="\t")
    args = parser.parse_args()

    data = load_data(args.train_path, args.test_path, args.data_sep, args.test_size, False, args.seed)

    results = []
    for file in tqdm(sorted(os.listdir(args.recs_dir))):
        if not file.endswith(".csv"):
            continue
        if not file.startswith("bprmf_"):
            continue

        full_path = os.path.join(args.recs_dir, file)
        recommender, strategy, params, config = parse_filename(file)
        scores = evaluate_file(full_path, data)
        results.append((file, (recommender, strategy, params, config, scores)))

    dataset_folder = os.path.relpath(args.recs_dir, "results/recommendations")
    output_path = os.path.join(args.output_dir, dataset_folder)
    os.makedirs(output_path, exist_ok=True)

    file = export_results_to_csv(results, output_path)
    print(f"\n✅ Resultados exportados a: {file}")


def main2():
    df = pd.read_csv("results/recommendations//NewYork_10/bprmf_NY_10_evaluation_results.csv")
    print(df[df.precision > 0.1][['filename', 'precision', 'recall', 'ndcg', 'batch_size']])


if __name__ == "__main__":
    main()
    main2()
