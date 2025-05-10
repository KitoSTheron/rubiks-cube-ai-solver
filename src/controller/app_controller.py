from model.data_model import DataModel
from view.main_window import MainWindow
import random
import threading
import copy

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
    
    def solve_cube(self, max_runtime=300):
        """
        Solve the cube using reinforcement learning with real-time visualization
        
        Args:
            max_runtime: Maximum runtime in seconds (default: 300s/5min)
        """
        print(f"Starting Rubik's Cube solver with {max_runtime}s maximum runtime...")
        
        # Import the RL solver
        from solver.rl_solver import RLCubeSolver
        import threading
        
        def solve_thread():
            try:
                # Initialize the solver with the current cube state
                solver = RLCubeSolver()
                
                # Use solve method with specified max_runtime
                solution_moves = solver.solve(
                    self.model.cube, 
                    controller=self, 
                    max_runtime=max_runtime
                )
                
                # Check if a solution was found
                if not solution_moves:
                    print("No solution found or maximum steps/time reached")
                    return
                
                print(f"Solution found with {len(solution_moves)} moves")
                print("Cube solved successfully!")
                
            except Exception as e:
                print(f"Error solving cube: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Start the solving in a separate thread to keep the UI responsive
        solver_thread = threading.Thread(target=solve_thread)
        solver_thread.daemon = True
        solver_thread.start()
    
    def update_view(self):
        """Update the view with the current cube state"""
        # Check if we're in the main thread
        if threading.current_thread() is threading.main_thread():
            self.view.update_view(self.model.cube)
        else:
            # Schedule the update to happen in the main thread
            self.view.root.after(0, lambda: self.view.update_view(self.model.cube))