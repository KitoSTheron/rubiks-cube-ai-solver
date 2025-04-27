import unittest
from src.cube.model import Cube

class TestCubeModel(unittest.TestCase):

    def setUp(self):
        self.cube = Cube()

    def test_initial_state(self):
        expected_state = [
            ['W', 'W', 'W', 'W'],
            ['W', 'W', 'W', 'W'],
            ['W', 'W', 'W', 'W'],
            ['W', 'W', 'W', 'W']
        ]
        self.assertEqual(self.cube.get_state(), expected_state)

    def test_rotate_face(self):
        self.cube.rotate_face('U')
        # Add assertions to check the state after rotating the upper face
        # This will depend on the implementation of the Cube class

    def test_get_state(self):
        self.cube.rotate_face('U')
        state_after_rotation = self.cube.get_state()
        # Add assertions to verify the state after rotation

if __name__ == '__main__':
    unittest.main()