from src.recommenders import Recommender
from src.utils import Recommendation
from src.data import Data
from src.strategies import Strategy
from typing import Dict, List, Tuple


class NeuralNetworkRecommender(Recommender):
    """
    Neural Network Recommender System using a feedforward neural network.
    
    Attributes:
        data (Data): user-item interaction data
        model (nn.Module): PyTorch model for the recommender system
    """
