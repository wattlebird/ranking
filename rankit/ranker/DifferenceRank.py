import numpy as np
from BaseRank import BaseRank

class DifferenceRank(BaseRank):
    """docstring for DifferenceRank"""
    def __init__(self, *args, **kwargs):
        super(DifferenceRank, self).__init__(*args, **kwargs)

    def rate(self, K):
        return K.sum(axis=1)/K.shape[0]
