import torch
from .basealgorithm import BaseOptimizer

from .fedavg import FedavgOptimizer



class FedlexOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(FedlexOptimizer, self).__init__(params=params, **kwargs)