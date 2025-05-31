from src.recommenders.basic_recommenders import Recommender
from src.utils.utils import Recommendation
from src.datamodule.data import Data
from src.utils.strategies import Strategy
from typing import Dict, List, Tuple


class NeuralNetworkRecommender(Recommender):
    """
    Neural Network Recommender System using a Multi-Layer Perceptron (MLP).
    
    Attributes:
        data (Data): user-item interaction data
    """
