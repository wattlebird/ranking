from numpy.linalg import solve

class DefaultSolver(object):
    """docstring for DefaultSolver"""
    def __init__(self, arg):
        super(DefaultSolver, self).__init__()
        self.arg = arg

    @classmethod
    def solve(cls, A, b):
        return solve(A, b)
