# RecommenderSystems

## Project Description
This project implements a modular Python library for building, training, and evaluating recommendation systems. It allows these models to be applied to various user-item datasets, supports automated evaluations using multiple metrics (accuracy, novelty, diversity), and enables performance comparison across recommendation systems.

As part of the project, we have implemented several recommendation algorithms, including:
- Popularity: Top-n most popular items.
- Random: Random sampling from the candidate pool.
- KNN (User/Item): Neighborhood-based Colaborative Filtering with cosine or Pearson similarity.
- Matrix Factorization (MF): Embedding-based MF trained with MSE loss.
- BPRMF: Bayesian Personalized Ranking objective over MF.
- MLP: Neural network on concatenated user/item embeddings.
- GNN: Graph-based propagation on user–item bipartite graph.

As well as several evaluation metrics, including:
- Precision: Fraction of recommended items that are relevant.
- Recall: Fraction of relevant items that are recommended.
- NDCG: Normalized Discounted Cumulative Gain.
- EPC: Expected Popularity Coverage.
- Gini: Gini coefficient of the recommended items.
- Aggregate Diversity: Average pairwise diversity of recommended items.

All methods share a common interface and can be easily run via command line or Python scripts. The library is designed to be extensible, allowing for the addition of new recommendation algorithms and evaluation metrics.

## Installation guide
1. Clone this repository:
```bash
git clone https://github.com/CataRVS/RecommenderSystems.git
cd RecommenderSystems
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Linux or Maxc, use:
source venv/bin/activate
# On Windows, use:
venv\Scripts\activate
```

3. Install the required Python packages:
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage guide
### Prepare data
To use the library, you need to prepare your dataset in a specific format. The library expects data in the form of user-item interactions, typically in a CSV format with four columns: `user_id`, `item_id`, `rating`, and `timestamp`.
### Usage via command line
#### Training a recommendation model
To train a recommendation model and generate recommendations, use the `main_recommend.py` script. This script allows configuration of the model, data, and recommendation strategies using flags. Example usage:
```bash
python -m src.main.main_recommend --recommender knn_user --data_path_train data/ml-100k/u1.base
```
##### Available flags:
| Flag                     | Description                                                                                            | Default Value              |
| ------------------------ | ------------------------------------------------------------------------------------------------------ | -------------------------- |
| `--recommender`          | Recommendation algorithm: `popularity`, `random`, `knn_user`, `knn_item`, `mf`, `bprmf`, `gnn`, `mlp`. | **Required**               |
| `--n_items_to_recommend` | Number of items to recommend per user.                                                                 | `10`                       |
| `--data_path_train`      | Path to the training data file (CSV/TSV).                                                              | **Required**               |
| `--data_path_test`       | Path to the test data file or `none` to split randomly.                                                | `none`                     |
| `--test_size`            | Proportion of data to use for testing (if `data_path_test` is `none`).                                 | `0.2`                      |
| `--sep`                  | Field separator in the data files.                                                                     | `\t`                       |
| `--ignore_first_line`    | Ignore the first line of the data files.                                                               | `False`                    |
| `--strategy`             | Recommendation strategy: `exclude_seen` or `no_filtering`.                                             | `exclude_seen`             |
| `--k`                    | Number of neighbors for KNN recommenders.                                                              | `5`                        |
| `--threshold`            | Threshold for KNN recommenders.                                                                        | `1.0`                      |
| `--similarity`           | Similarity metric for KNN recommenders: `cosine` or `pearson`.                                         | `pearson`                  |
| `--n_factors`            | Latent dimension for embeddings (MF/BPRMF/MLP/GNN).                                                    | `20`                       |
| `--lr`                   | Learning rate (MF/BPRMF/GNN).                                                                          | `0.01`                     |
| `--reg`                  | L2 regularization coefficient (MF/BPRMF/MLP/GNN).                                                      | `0.1`                      |
| `--n_epochs`             | Number of training epochs (MF/BPRMF/MLP/GNN).                                                          | `10`                       |
| `--batch_size`           | Mini-batch size (MF/BPRMF/MLP/GNN).                                                                    | `4096`                     |
| `--device`               | `cpu` or `cuda` (automatically detected if not specified).                                             | `auto`                     |
| `--hidden_dims`          | Hidden layer dimensions for MLP, e.g., `64 32`.                                                        | `64 32`                    |
| `--n_layers`             | Number of convolution layers for GNN.                                                                  | `3`                        |
| `--seed`                 | Random seed for reproducibility.                                                                       | `42`                       |
| `--save_path`            | Path to save the generated recommendations.                                                            | `results/recommendations/` |


#### Evaluating Recommendations 
To evaluate the generated recommendations, use the `main_evaluate.py` script. This script allows calculation of metrics such as precision, recall, NDCG, and others. Example usage:
```bash
python -m src.main.main_evaluate --metric precision --recommendations_path results/recommendations/knn_user.csv --data_path_train data/ml-100k/u1.test
```
##### Available flags:
| Flag                       | Description                                                                             | Default Value                             |
| -------------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------- |
| `--metric`                 | Evaluation metric: `precision`, `recall`, `ndcg`, `epc`, `gini`, `aggregate_diversity`. | **Required**                              |
| `--recommendations_path`   | Path to the file with generated recommendations.                                        | **Required**                              |
| `--data_path_train`        | Path to the training data file.                                                         | **Required**                              |
| `--data_path_test`         | Path to the test data file or `none` to split randomly.                                 | `none`                                    |
| `--test_size`              | Proportion of data to use for testing (if `data_path_test` is `none`).                  | `0.2`                                     |
| `--sep_recs`               | Separator used in the recommendations file.                                             | `,`                                       |
| `--sep_data`               | Separator used in the data files.                                                       | `\t`                                      |
| `--ignore_first_line_recs` | Ignore the first line of the recommendations file.                                      | `False`                                   |
| `--ignore_first_line_data` | Ignore the first line of the data files.                                                | `False`                                   |
| `--seed`                   | Random seed for reproducibility.                                                        | `42`                                      |


### Usage via Pipelines scripts
To use the library via pipelines, you can run the scripts located in the `src/pipelines/` directory. These scripts permit to automate the training and evaluation processes, including data processing, model training, and evaluation plotting.
#### Data Processing
To process the data, you can use the `kcore.py` script. This script allows you to filter the dataset based on user and item activity, which is useful for preparing the data for training and evaluation. To use it, open the script and modify the parameters as needed, and then run it:
Example usage:
```bash
python -m src.pipelines.data_processing.kcore
```
#### Training
To train a recommendation model, you can use the `train_lot.py` script. This script allows you to specify the recommender algorithm and select a number of parameters to perform a grid search over them. It will save the trained model and the recommendations to the specified path. To use it, open the script and modify the parameters as needed, and then run it:
Example usage:
```bash
python -m src.pipelines.train.train_lot
```

#### Evaluation
To evaluate the recommendations generated by the trained model, you can use the `evaluate_lot.py` script. This script evaluates the recommendations made by a specific recommender algorithm and calculates all the metrics. It will save the results of all the metrics in a csv file. To use it, open the script and modify the parameters as needed, and then run it:
```bash
python -m src.pipelines.evaluate.evaluate_lot
```

#### Plotting
To visualize the evaluation results, you can use the `plot_lot.py` script. This script generates plots for the evaluation metrics and saves them to the specified directory. It will also print tables of the best results for each recommender algorithm and a comparison of the best results across all algorithms. This is useful for comparing the performance of different recommendation algorithms visually. To use it, open the script and modify the parameters as needed, and then run it:
```bash
python -m src.pipelines.evaluate.plot_lot
```


## Project structure
```
RecommenderSystems/
├───commands/               # Example commands for running the library
│   ├───evaluation/             # Commands for evaluating recommendations
│   └───train/                  # Commands for training recommendation models
│
├───data/                   # Datasets used for training and evaluation
│   ├───ml-100k/                # Original MovieLens 100k dataset
│   └───NewYork/                # Foursquare New York City POI dataset
│
├───results/                # Results of the evaluations and recommendations
│   ├───graphs/                 # Graphs obtained from the evaluations
│   │   ├───ml-100k/
│   │   └───NewYork_10/
│   ├───metrics/                # Metrics obtained from the evaluations
│   │   ├───ml-100k/
│   │   └───NewYork_10/
│   └───recommendations/        # Recommendations generated by the models
│       ├───ml-100k/
│       └───NewYork_10/
│
├───src/                    # Source code of the library
│   ├───datamodule/             # Data loading and processing modules
│   │       __init__.py
│   │       data.py
│   ├───evaluation/             # Evaluation modules for recommendation systems
│   │       __init__.py
│   │       evaluation.py
│   ├───main/                   # Main scripts for training and evaluating models via command line
│   │       main_evaluate.py
│   │       main_recommend.py
│   ├───pipelines/              # Pipelines for training and evaluating recommendation systems
│   │   ├───data_processing/        # Data processing scripts
│   │   │       kcore.py
│   │   ├───evaluate/               # Evaluation and ploting scripts
│   │   │       evaluate_lot.py
│   │   │       plot_lot.py
│   │   └───train/                  # Training scripts
│   │           train_lot.py
│   ├───recommenders/           # Implementations of various recommendation algorithms
│   │       __init__.py
│   │       basic_recommenders.py
│   │       knn.py
│   │       matrix_factorisation.py
│   │       neural_networks.py
│   └───utils/                  # Utility functions and classes
│           __init__.py
│           datasets.py
│           similarities.py
│           strategies.py
│           utils.py
├───.gitignore              # Git ignore file
├───README.md               # Project documentation
└───requirements.txt        # Python package dependencies
```


