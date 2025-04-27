import unittest
from src.cube.solver import Solver

class TestSolver(unittest.TestCase):

    def setUp(self):
        self.solver = Solver()

    def test_is_solved(self):
        self.assertTrue(self.solver.is_solved())

    def test_solve(self):
        # Assuming we have a method to scramble the cube
        self.solver.scramble()
        solution = self.solver.solve()
        self.assertTrue(self.solver.is_solved())

if __name__ == '__main__':
    unittest.main()