from rankit.solver import DefaultSolver
from BaseRank import BaseRank


class ColleyRank(BaseRank):
    """docstring for ColleyRank"""

    def __init__(self, itemlist):
        super(ColleyRank, self).__init__(itemlist)

    def rate(self, C, b):
        return DefaultSolver.solve(C, b)


