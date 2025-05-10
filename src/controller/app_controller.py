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
    
    def solve_cube(self):
        """Solve the cube using reinforcement learning with real-time visualization"""
        print("Starting Rubik's Cube solver with reinforcement learning...")
        
        # Import the RL solver
        from solver.rl_solver import RLCubeSolver
        import threading
        
        def update_ui():
            """Update the UI from the main thread"""
            self.update_view()
            self.view.root.update_idletasks()
        
        def solve_thread():
            try:
                # Initialize the solver with the current cube state
                solver = RLCubeSolver()
                
                # Use only the standard solve method - don't pass the ui_callback
                solution_moves = solver.solve(self.model.cube, controller=self)
                
                # Final update when done
                self.view.root.after(0, update_ui)
                
                # Check if a solution was found
                if not solution_moves:
                    print("No solution found or maximum steps reached")
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
    
    def process_next_solving_step(self):
        """Process one step of the solving algorithm, then schedule the next step"""
        if self.current_step >= self.max_steps or self.solver._is_solved(self.current_state):
            if self.solver._is_solved(self.current_state):
                print(f"Solution found in {self.current_step} steps!")
            else:
                print(f"Maximum steps reached without solution.")
            return
        
        # Perform one step of the solving process
        # This is a simplified version - you'll need to adapt the complex logic from the solver
        action_scores = []
        for action in self.solver.actions:
            face, direction = action
            new_state = self.solver._apply_action(self.current_state, face, direction)
            score = self.solver._evaluate_state(new_state)
            action_scores.append((action, new_state, score))
        
        # Sort by score and select the best action
        action_scores.sort(key=lambda x: x[2], reverse=True)
        best_action, best_state, best_score = action_scores[0]
        face, direction = best_action
        
        # Apply the action
        self.model.cube = copy.deepcopy(self.current_state)
        self.model.rotate_face(face, direction)
        self.update_view()
        
        # Update state
        self.current_state = best_state
        self.solution_moves.append(best_action)
        self.current_step += 1
        
        # Update best state if needed
        if best_score > self.best_score_overall:
            self.best_score_overall = best_score
            self.best_state_overall = copy.deepcopy(self.current_state)
        
        # Schedule the next step with a delay
        self.view.root.after(200, self.process_next_solving_step)
    
    def update_view(self):
        """Update the view with the current cube state"""
        # Check if we're in the main thread
        if threading.current_thread() is threading.main_thread():
            self.view.update_view(self.model.cube)
        else:
            # Schedule the update to happen in the main thread
            self.view.root.after(0, lambda: self.view.update_view(self.model.cube))