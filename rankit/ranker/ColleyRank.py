from rankit.solver import DefaultSolver, InsufficientRankSolver
from BaseRank import BaseRank


class ColleyRank(BaseRank):
    """docstring for ColleyRank"""

    def __init__(self, *args, **kwargs):
        self.solver = DefaultSolver()
        super(ColleyRank, self).__init__(*args, **kwargs)

    def rate(self, C, b):
        solver = self.solver
        return solver.solve(C, b)


class MasseyRank(BaseRank):
    def __init__(self, *args, **kwargs):
        self.solver = InsufficientRankSolver(force=1)
        super(MasseyRank, self).__init__(*args, **kwargs)

    def rate(self, M, b):
        solver = self.solver
        return solver.solve(M, b)
