import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_all_csv_from_folder(folder_path: str) -> pd.DataFrame:
    """
    Load and concatenate all CSV files in a folder into a single DataFrame.

    Parameters:
        folder_path (str): Path to the folder containing CSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame containing data from all CSV files.
    """
    # If the folder does not exist, print a message and exit
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        exit(1)

    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # If no CSV files found, print a message and exit
    if not csv_files:
        print(f"No CSV files found in the folder {folder_path}.")
        exit(1)

    dfs = []

    for file in csv_files:
        # Read each CSV file and append a new column with the source file name
        full_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(full_path)
        except pd.errors.EmptyDataError:
            print(f"Warning: {file} is empty and will be skipped.")
            continue
        except pd.errors.ParserError:
            print(f"Warning: {file} could not be parsed and will be skipped.")
            continue
        except Exception:
            print(f"Error reading {file} and will be skipped.")
            continue
        df['source_file'] = file
        dfs.append(df)

    # If there are no valid DataFrames, print a message and exit
    if not dfs:
        print(f"No valid CSV files found in the folder {folder_path}.")
        exit(1)

    return pd.concat(dfs, ignore_index=True)


def plot_precision_vs_diversity(df: pd.DataFrame, graphs_path: str):
    """
    Plot trade-off between precision and aggregate diversity.

    Parameters:
        df (pd.DataFrame): DataFrame containing evaluation results with 'recommender',
            'precision', and 'aggregate_diversity' columns.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='precision', y='aggregate_diversity', hue='recommender')
    plt.title('Precision vs. Aggregate Diversity')
    plt.xlabel('Precision')
    plt.ylabel('Aggregate Diversity')
    plt.legend(title='Recommender')
    plt.tight_layout()
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(graphs_path, 'precision_vs_diversity.png'))


def plot_knn_precision_by_k(df: pd.DataFrame, graphs_path: str):
    """
    Plot KNN precision as a function of k and similarity type.

    Parameters:
        df (pd.DataFrame): DataFrame containing evaluation results with 'recommender',
            'k', 'similarity', and 'precision' columns.
        graphs_path (str): Path to save the generated plots.
    """
    df_knn = df[df['recommender'].str.contains('knn')]
    if df_knn.empty:
        print("No KNN data found in the DataFrame.")
        return
    df_knn = df_knn.sort_values(by='precision', ascending=False)
    print("\nKNN results:")
    print(df_knn[[
        'recommender', 'k', 'similarity', 'precision', 'recall', 'ndcg', 'epc', 'gini',
        'aggregate_diversity'
    ]])
    df_knn = df_knn[['recommender', 'k', 'similarity', 'precision']]

    # Plot them all together
    plt.figure(figsize=(10, 6))
    for (recommender, similarity), group in df_knn.groupby(['recommender', 'similarity']):
        sns.lineplot(
            data=group, x='k', y='precision', label=f"{recommender} - {similarity}", marker='o'
        )
    plt.title('KNN: Precision by k and Similarity Type')
    plt.xlabel('k')
    plt.ylabel('Precision')
    plt.legend(title='Recommender - Similarity')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(graphs_path, 'knn_precision_by_k.png'))


def plot_bprmf_precision_by_factors(df: pd.DataFrame, graphs_path: str):
    """
    Plot BPRMF precision as a function of the number of latent factors.

    Parameters:
        df (pd.DataFrame): DataFrame containing evaluation results with 'recommender',
            'n_factors', and 'precision' columns.
    """
    df_bprmf = df[df['recommender'] == 'bprmf']
    if df_bprmf.empty:
        print("No BPRMF data found in the DataFrame.")
        return
    df_bprmf = df_bprmf.sort_values(by='precision', ascending=False)
    print("\nTop 5 BPRMF configurations based on precision:")
    print(df_bprmf.head(5)[[
        'recommender', 'n_factors', 'regularization', 'learning_rate', 'epochs', 'batch_size',
        'precision', 'recall', 'ndcg', 'epc', 'gini', 'aggregate_diversity',
    ]])
    if 'n_factors' not in df_bprmf.columns:
        print("n_factors column is missing in BPRMF data.")
        return
    try:
        plt.figure(figsize=(6, 4))
        sns.lineplot(data=df_bprmf, x='n_factors', y='precision', marker='o')
        plt.title('BPRMF: Precision by Number of Latent Factors')
        plt.xlabel('Number of Factors')
        plt.ylabel('Precision')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(graphs_path, 'bprmf_precision_by_factors.png'))
    except KeyError and ValueError:
        print("n_factors column is missing in BPRMF data or contains invalid values.")
        return


def plot_mf_precision_by_factors(df: pd.DataFrame, graphs_path: str):
    """
    Plot MF precision as a function of the number of latent factors.

    Parameters:
        df (pd.DataFrame): DataFrame containing evaluation results with 'recommender',
            'n_factors', and 'precision' columns.
    """
    df_mf = df[df['recommender'] == 'mf']
    if df_mf.empty:
        print("No MF data found in the DataFrame.")
        return
    elif 'n_factors' not in df_mf.columns:
        print("n_factors column is missing in MF data.")
        return

    df_mf = df_mf.sort_values(by='precision', ascending=False)
    print("\nTop 5 MF configurations based on precision:")
    print(df_mf.head(5)[[
        'recommender', 'n_factors', 'regularization', 'learning_rate', 'epochs', 'batch_size',
        'precision', 'recall', 'ndcg', 'epc', 'gini', 'aggregate_diversity'
    ]])

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df_mf, x='n_factors', y='precision', marker='o')
    plt.title('MF: Precision by Number of Latent Factors')
    plt.xlabel('Number of Factors')
    plt.ylabel('Precision')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(graphs_path, 'mf_precision_by_factors.png'))


def show_mlp_parameters(df: pd.DataFrame):
    """
    Show top 5 MLP configurations based on precision.

    Parameters:
        df (pd.DataFrame): DataFrame containing all evaluation results.
    """
    df_mlp = df[df['recommender'] == 'mlp']
    if df_mlp.empty:
        print("No MLP data found in the DataFrame.")
        return

    df_mlp = df_mlp.sort_values(by='precision', ascending=False)
    print("\nTop 5 MLP configurations based on precision:")
    print(df_mlp.head(5)[[
        'recommender', 'n_factors', 'regularization', 'learning_rate', 'hidden_dims', 'epochs',
        'batch_size', 'precision', 'recall', 'ndcg', 'epc', 'gini', 'aggregate_diversity'
    ]])


def show_gnn_parameters(df: pd.DataFrame):
    """
    Show top 5 GNN configurations based on precision.

    Parameters:
        df (pd.DataFrame): DataFrame containing all evaluation results.
    """
    df_gnn = df[df['recommender'] == 'gnn']
    if df_gnn.empty:
        print("No GNN data found in the DataFrame.")
        return

    df_gnn = df_gnn.sort_values(by='precision', ascending=False)
    print("\nTop 5 GNN configurations based on precision:")
    print(df_gnn.head(5)[[
        'recommender', 'n_factors', 'regularization', 'learning_rate', 'n_layers', 'epochs',
        'batch_size', 'precision', 'recall', 'ndcg', 'epc', 'gini', 'aggregate_diversity'
    ]])


def plot_best_total(df: pd.DataFrame, graphs_path: str):
    """
    Plot and show best evaluation results for each recommender.

    Parameters:
        df (pd.DataFrame): DataFrame containing evaluation results with 'recommender',
            'precision', 'recall', 'ndcg', 'epc', 'gini', and 'aggregate_diversity' columns.
        graphs_path (str): Path to save the generated plots.
    """
    idx = df.groupby("recommender")["precision"].idxmax()
    df_best = df.loc[idx].set_index("recommender")

    if df_best.empty:
        print("No best results found in the DataFrame.")
        return

    metrics = ["precision", "recall", "ndcg", "epc", "gini", "aggregate_diversity"]

    print("\nBest results for each recommender:")
    print(df_best[metrics])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    recommenders = df_best.index

    for ax, metric in zip(axes.flatten(), metrics):
        df_best[metric].plot(kind='bar', ax=ax, title=metric.capitalize())
        ax.set_xticklabels(recommenders, rotation=45)
        ax.set_ylabel(metric.capitalize())
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle("Best Precision Results for Recommenders", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_path, "best_results.png"))


def main(metrics_path: str, graphs_path: str):
    """
    Main function to load evaluation results and generate plots.

    Parameters:
        metrics_path (str): Path to the folder containing evaluation results CSV files.
        graphs_path (str): Path to save the generated plots.
    """
    df_all = load_all_csv_from_folder(metrics_path)

    # Ensure the output directory exists
    os.makedirs(graphs_path, exist_ok=True)

    plot_knn_precision_by_k(df_all, graphs_path)
    plot_bprmf_precision_by_factors(df_all, graphs_path)
    plot_mf_precision_by_factors(df_all, graphs_path)
    show_mlp_parameters(df_all)
    show_gnn_parameters(df_all)
    plot_best_total(df_all, graphs_path)


if __name__ == "__main__":
    ########## CONFIGURATION ##########
    metrics_path = "results/metrics/NewYork"
    graphs_path = "results/graphs/NewYork"
    metrics_path = "results/metrics/ml-100k"
    graphs_path = "results/graphs/ml-100k"
    main(metrics_path, graphs_path)
