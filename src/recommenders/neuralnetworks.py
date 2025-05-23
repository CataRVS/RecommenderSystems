from src.recommenders.basic_recommenders import Recommender
from src.utils.utils import Recommendation
from src.datamodule.data import Data
from src.utils.strategies import Strategy
from typing import Dict, List, Tuple


class NeuralNetworkRecommender(Recommender):
    """
    Neural Network Recommender System using a feedforward neural network.
    
    Attributes:
        data (Data): user-item interaction data
        model (nn.Module): PyTorch model for the recommender system
    """
