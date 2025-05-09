from model.data_model import DataModel
from view.main_window import MainWindow
import random

class AppController:
    """
    Application controller.
    Connects the model and view components.
    """
    
    def __init__(self):
        """Initialize the application controller"""
        self.model = DataModel()
        self.view = MainWindow(self)
    
    def run(self):
        """Start the application"""
        self.view.run()
    
    def update_model(self, key, value):
        """
        Update the model with new data
        
        Args:
            key: The data key
            value: The data value
        """
        self.model.set_data(key, value)
        # Update the view with the new data
        self.view.update_view(self.model.get_data())
    
    def rotate_face(self, face, direction):
        """
        Rotate a face of the cube
        
        Args:
            face: One of 'U', 'D', 'L', 'R', 'F', 'B'
            direction: 'clockwise', 'counterclockwise', or 'double'
        """
        self.model.rotate_face(face, direction)
        # Update the view after rotation
        self.update_view()
    
    def reset_cube(self):
        """Reset the cube to its solved state"""
        # Reset cube to its initial solved state by copying from the solved state
        import copy
        self.model.cube = copy.deepcopy(self.model.solved)
        
        print("Cube has been reset to solved state")
        
        # Update the view after resetting
        self.update_view()
    
    def scramble_cube(self):
        """Randomly scramble the cube"""
        # Perform 20 random moves
        faces = ['U', 'D', 'L', 'R', 'F', 'B']
        directions = ['clockwise', 'counterclockwise', 'double']
        
        print("Scrambling cube...")
        for _ in range(20):
            face = random.choice(faces)
            direction = random.choice(directions)
            self.model.rotate_face(face, direction)
        
        print("Cube scrambled")
        # Update the view after scrambling
        self.update_view()
    
    def solve_cube(self):
        """Solve the cube"""
        # This would be implemented later
        print("Solve cube requested")
        # After solving, update the view
        self.update_view()
    
    def update_view(self):
        """Update the view with the current cube state"""
        self.view.update_view(self.model.cube)