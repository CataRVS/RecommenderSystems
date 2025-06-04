# RecommenderSystems

## Project Description


## Installation guide

## Usage guide
### Dowload the project
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
To generate recommendatios, you have the following options:
- Run the `main_recommend.py` script contained in the `main` folder with the desired arguments:
- Run the `train_lot.py` script contained in the `pipelines/train` folder after changing the indicated parameters in the file.
```bash
python -m src.main data/ml-100k/u.data
```
