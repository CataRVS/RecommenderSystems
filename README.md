# RecommenderSystems

## Project Description
This project implements a modular Python library for building, training, and evaluating recommendation systems. It includes models ranging from simple approaches like popularity and k-NN to advanced techniques such as matrix factorization, MLP, and Graph Neural Networks (GNNs).

The library allows these models to be applied to various user-item datasets, supports automated evaluations using multiple metrics (accuracy, novelty, diversity), and enables performance comparison across recommendation systems.

## Usage guide
### Download the project
1. Clone the repository:
```bash
git clone https://github.com/CataRVS/RecommenderSystems.git
```

2. Navigate to the project directory:
```bash
cd RecommenderSystems
```

3. Install the required Python packages:
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Generate recommendations
To generate recommendations, you have the following options:
- Run the `main_recommend.py` script contained in the `src/main` folder with the desired arguments:
```bash
python -m src.main.main_recommend **kwargs
```

- Run the `train_lot.py` script contained in the `src/pipelines/train` folder after changing the indicated parameters in the file.
```bash
python -m src.pipelines.train.train_lot data/ml-100k/u.data
```

### Evaluate recommendations
To evaluate recommendations, you have the following options:
- Run the `main_evaluate.py` script contained in the `src/main` folder with the desired arguments:
```bash
python -m src.main.main_evaluate **kwargs
```
- Run the `evaluate_lot.py` script contained in the `src/pipelines/evaluate` folder after changing the indicated parameters in the file.
```bash
python -m src.pipelines.evaluate.evaluate_lot data/ml-100k/u.data
```

### Generate tables and plots
To generate tables and plots, you can run the following script after changing the indicated parameters in the file:
```bash
python -m src.pipelines.evaluate.plot_lot
```
