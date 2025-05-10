import numpy as np
import tensorflow as tf
import random
import copy

class RLCubeSolver:
    """
    Reinforcement Learning based solver for the Rubik's Cube
    
    This solver uses Deep Q-Learning to find a solution sequence
    for a scrambled Rubik's Cube
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the RL solver
        
        Args:
            model_path: Optional path to a saved model
        """
        # Define possible actions first, before any other method calls
        self.actions = []
        for face in ['U', 'D', 'L', 'R', 'F', 'B']:
            for direction in ['clockwise', 'counterclockwise', 'double']:
                self.actions.append((face, direction))
        
        self.max_steps = 100  # Maximum number of steps to attempt
        # Now load the model after actions is defined
        self.model = self._load_or_create_model(model_path)
    
    def _load_or_create_model(self, model_path):
        """Load or create a new TensorFlow model"""
        try:
            if model_path:
                print(f"Loading model from {model_path}")
                # Replace with your TensorFlow model loading code
                model = tf.keras.models.load_model(model_path)
                return model
            else:
                print("Creating new model")
                # This is a placeholder for your actual model architecture
                model = self._create_model()
                return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Creating new model instead")
            return self._create_model()
    
    def _create_model(self):
        """Create a new TensorFlow model for cube solving"""
        # Ensure actions are defined
        if not hasattr(self, 'actions'):
            # Define actions if not already defined
            self.actions = []
            for face in ['U', 'D', 'L', 'R', 'F', 'B']:
                for direction in ['clockwise', 'counterclockwise', 'double']:
                    self.actions.append((face, direction))
        
        # Example of a simple DQN network
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(324,)),  # Explicit input layer
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(len(self.actions))  # Q-values for each action
            ])
            model.compile(optimizer='adam', loss='mse')
            return model
        except Exception as e:
            print(f"Error creating model: {e}")
            # Return a simple placeholder model if there's an error
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(324,)),
                tf.keras.layers.Dense(10, activation='relu'),
                tf.keras.layers.Dense(len(self.actions))
            ])
            model.compile(optimizer='adam', loss='mse')
            return model
    
    def _state_to_features(self, cube_state):
        """Convert the cube state to features for the model"""
        # This is a placeholder - implement your feature extraction here
        # You'll need to convert the cube representation to a format suitable for the NN
        features = []
        # Example of a simple one-hot encoding approach:
        color_map = {'y': 0, 'b': 1, 'r': 2, 'o': 3, 'g': 4, 'w': 5}
        
        for face in cube_state:
            for row in face:
                for sticker in row:
                    color = sticker[0]  # Get the color code
                    # One-hot encode the color
                    one_hot = [0] * 6
                    one_hot[color_map[color]] = 1
                    features.extend(one_hot)
        
        return np.array([features])
    
    def _is_solved(self, cube_state):
        """Check if the cube is solved"""
        # A cube is solved when all stickers on each face have the same color
        for face in cube_state:
            first_color = face[0][0][0]
            for row in face:
                for sticker in row:
                    if sticker[0] != first_color:
                        return False
        return True
    
    def _apply_action(self, cube_state, face, direction):
        """
        Apply a move to the cube state and return the new state
        
        Args:
            cube_state: Current cube state
            face: One of 'U', 'D', 'L', 'R', 'F', 'B'
            direction: 'clockwise', 'counterclockwise', or 'double'
            
        Returns:
            New cube state after applying the move
        """
        # Create a temporary DataModel to apply the move
        from model.data_model import DataModel
        temp_model = DataModel()
        
        # Replace the cube state with our current state
        temp_model.cube = copy.deepcopy(cube_state)
        
        # Apply the move
        temp_model.rotate_face(face, direction)
        
        # Return the new state
        return temp_model.cube

    def _apply_algorithm(self, cube_state, algorithm):
        """
        Apply a sequence of moves (an algorithm) to the cube
        
        Args:
            cube_state: Current cube state
            algorithm: List of (face, direction) tuples
            
        Returns:
            New cube state after applying the algorithm
        """
        # Create a temporary DataModel to apply the moves
        from model.data_model import DataModel
        temp_model = DataModel()
        
        # Replace the cube state with our current state
        temp_model.cube = copy.deepcopy(cube_state)
        
        # Apply each move in the algorithm
        for face, direction in algorithm:
            temp_model.rotate_face(face, direction)
        
        # Return the new state
        return temp_model.cube

    def _get_common_algorithms(self):
        """
        Return a list of common algorithms for Rubik's Cube
        
        Returns:
            List of algorithm tuples (name, moves)
        """
        algorithms = [
            # Basic algorithms
            ("R Move", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise")]),
            ("L Move", [("L", "clockwise"), ("U", "clockwise"), ("L", "counterclockwise"), ("U", "counterclockwise")]),
            ("F Move", [("F", "clockwise"), ("U", "clockwise"), ("F", "counterclockwise"), ("U", "counterclockwise")]),
            ("B Move", [("B", "clockwise"), ("U", "clockwise"), ("B", "counterclockwise"), ("U", "counterclockwise")]),
            ("U Move", [("U", "clockwise"), ("R", "clockwise"), ("U", "counterclockwise"), ("R", "counterclockwise")]),
            ("D Move", [("D", "clockwise"), ("L", "clockwise"), ("D", "counterclockwise"), ("L", "counterclockwise")]),

            # OLL Algorithms (Orient Last Layer)
            ("OLL - Dot Case", [("F", "clockwise"), ("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise"), ("F", "counterclockwise")]),
            ("OLL - T Shape", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise"), ("R", "counterclockwise"), ("F", "clockwise"), ("R", "F", "counterclockwise")]),
            ("OLL - L Shape", [("F", "clockwise"), ("R", "U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise"), ("F", "counterclockwise")]),
            ("OLL - Line", [("F", "clockwise"), ("R", "U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise"), ("F", "counterclockwise")]),
            ("OLL - Cross", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "clockwise"), ("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise")]),

            # PLL Algorithms (Permute Last Layer)
            ("PLL - U Perm", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "clockwise"), ("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise")]),
            ("PLL - H Perm", [("M", "clockwise"), ("U", "clockwise"), ("M", "clockwise"), ("U", "clockwise"), ("M", "clockwise"), ("U", "clockwise"), ("M", "clockwise"), ("U", "clockwise")]),
            ("PLL - Z Perm", [("M", "clockwise"), ("U", "clockwise"), ("M", "clockwise"), ("U", "counterclockwise"), ("M", "clockwise"), ("U", "clockwise"), ("M", "clockwise"), ("U", "counterclockwise")]),
            ("PLL - A Perm", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("F", "clockwise"), ("R", "clockwise"), ("U", "counterclockwise"), ("R", "counterclockwise"), ("F", "counterclockwise")]),
            ("PLL - E Perm", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise"), ("R", "clockwise"), ("F", "clockwise"), ("R", "counterclockwise"), ("F", "counterclockwise")]),

            # F2L Algorithms (First Two Layers)
            ("F2L - Pair Insert", [("U", "clockwise"), ("R", "clockwise"), ("U", "counterclockwise"), ("R", "counterclockwise"), ("U", "counterclockwise"), ("F", "clockwise"), ("U", "clockwise"), ("F", "counterclockwise")]),
            ("F2L - Edge Flip", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "clockwise"), ("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise")]),
            ("F2L - Corner Insert", [("U", "clockwise"), ("R", "clockwise"), ("U", "counterclockwise"), ("R", "counterclockwise"), ("U", "clockwise"), ("F", "clockwise"), ("U", "counterclockwise"), ("F", "counterclockwise")]),

            # Equivalent algorithms for different orientations
            ("R Move (Front Facing)", [("F", "clockwise"), ("U", "clockwise"), ("F", "counterclockwise"), ("U", "counterclockwise")]),
            ("R Move (Left Facing)", [("L", "clockwise"), ("U", "clockwise"), ("L", "counterclockwise"), ("U", "counterclockwise")]),
            ("R Move (Back Facing)", [("B", "clockwise"), ("U", "clockwise"), ("B", "counterclockwise"), ("U", "counterclockwise")]),
            ("R Move (Right Facing)", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise")]),

            ("Sledgehammer (Front Facing)", [("F", "clockwise"), ("R", "clockwise"), ("F", "counterclockwise"), ("R", "counterclockwise")]),
            ("Sledgehammer (Left Facing)", [("L", "clockwise"), ("F", "clockwise"), ("L", "counterclockwise"), ("F", "counterclockwise")]),
            ("Sledgehammer (Back Facing)", [("B", "clockwise"), ("L", "clockwise"), ("B", "counterclockwise"), ("L", "counterclockwise")]),
            ("Sledgehammer (Right Facing)", [("R", "clockwise"), ("B", "clockwise"), ("R", "counterclockwise"), ("B", "counterclockwise")]),

            # Additional algorithms for advanced solving
            ("OLL - Fish Shape", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "clockwise"), ("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise")]),
            ("PLL - T Perm", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise"), ("R", "counterclockwise"), ("F", "clockwise"), ("R", "F", "counterclockwise")]),
            ("PLL - J Perm", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("F", "clockwise"), ("R", "clockwise"), ("U", "counterclockwise"), ("R", "counterclockwise"), ("F", "counterclockwise")]),
        ]
        return algorithms

    def _evaluate_state(self, cube_state):
        """Improved evaluation with more progressive scoring and memory of past good states"""
        # Add this check at the beginning of the method
        if cube_state is None:
            print("WARNING: Received None cube state in _evaluate_state")
            return -float('inf')  # Return very negative score for None states
        
        score = 0
        
        # --- WHITE CROSS DETECTION with HIGHER RETENTION VALUE ---
        bottom_face = 5  # White face
        front_face = 1   # Blue face
        right_face = 2   # Red face
        left_face = 3    # Orange face
        back_face = 4    # Green face
        top_face = 0     # Yellow face
        
        # Check white center
        if cube_state is not None and cube_state[bottom_face][1][1][0] == 'w':
            score += 5
            
            # Check white edges with MUCH higher values
            white_edge_positions = [(0, 1), (1, 0), (1, 2), (2, 1)]
            adjacent_faces = [front_face, left_face, right_face, back_face]
            adjacent_pos = [(2, 1), (2, 1), (2, 1), (2, 1)]
            expected_colors = ['b', 'o', 'r', 'g']
            
            white_edges_correct = 0
            
            # Track each correct edge piece separately with high value
            for i, pos in enumerate(white_edge_positions):
                row, col = pos
                adj_row, adj_col = adjacent_pos[i]
                adj_face = adjacent_faces[i]
                
                # White sticker in the correct position and orientation
                if cube_state[bottom_face][row][col][0] == 'w' and cube_state[adj_face][adj_row][adj_col][0] == expected_colors[i]:
                    score += 25  # Higher value for correct position and orientation
                    white_edges_correct += 1
            
            # Bonus for complete white cross
            if white_edges_correct == 4:
                score += 50  # Higher bonus (was 10)
                
                # Evaluate F2L (First Two Layers)
                f2l_edges_positions = [
                    (front_face, (1, 2), right_face, (1, 0)),
                    (front_face, (1, 0), left_face, (1, 2)),
                    (back_face, (1, 2), left_face, (1, 0)),
                    (back_face, (1, 0), right_face, (1, 2))
                ]
                expected_f2l_colors = [
                    ['b', 'r'],
                    ['b', 'o'],
                    ['g', 'o'],
                    ['g', 'r']
                ]

                f2l_edges_correct = 0

                for i, (face1, pos1, face2, pos2) in enumerate(f2l_edges_positions):
                    expected_colors = expected_f2l_colors[i]

                    # Check if the edge piece is in the correct position
                    if cube_state[face1][pos1[0]][pos1[1]][0] == expected_colors[0] and \
                       cube_state[face2][pos2[0]][pos2[1]][0] == expected_colors[1]:
                        score += 40  # Higher value for correct F2L edge position
                        f2l_edges_correct += 1

                # Bonus for completing F2L
                if f2l_edges_correct == 4:
                    score += 100  # Higher bonus for completing F2L

                # Evaluate OLL (Orient Last Layer)
                top_face_colors = [sticker[0] for row in cube_state[top_face] for sticker in row]
                if all(color == 'y' for color in top_face_colors):
                    score += 150  # Bonus for completing OLL

                # Evaluate PLL (Permute Last Layer)
                if self._is_solved(cube_state):
                    score += 200  # Bonus for solving the cube
        
        return score
    
    def _evaluate_white_cross(self, cube_state):
        """Evaluate just the white cross"""
        score = 0
        bottom_face = 5
        front_face = 1
        right_face = 2
        left_face = 3
        back_face = 4
        
        # Only score if white center is correct
        if cube_state[bottom_face][1][1][0] == 'w':
            white_edge_positions = [(0, 1), (1, 0), (1, 2), (2, 1)]
            adjacent_faces = [front_face, left_face, right_face, back_face]
            adjacent_pos = [(2, 1), (2, 1), (2, 1), (2, 1)]
            expected_colors = ['b', 'o', 'r', 'g']
            
            for i, pos in enumerate(white_edge_positions):
                row, col = pos
                adj_row, adj_col = adjacent_pos[i]
                adj_face = adjacent_faces[i]
                
                if cube_state[bottom_face][row][col][0] == 'w' and cube_state[adj_face][adj_row][adj_col][0] == expected_colors[i]:
                        score += 25
        
        return score

    def _get_state_hash(self, cube_state):
        """
        Convert a cube state to a hashable representation
        for detecting repeated states
        """
        # Simple approach: just join the color codes
        hash_str = ""
        for face in cube_state:
            for row in face:
                for sticker in row:
                    hash_str += sticker[0]
        return hash_str

    def _calculate_repeat_penalty(self, move_history, move):
        """
        Calculate a penalty score for repeating recent moves or patterns
        
        Args:
            move_history: List of previous moves
            move: The move being considered (face, direction)
        
        Returns:
            A penalty score (higher for repetitive moves)
        """
        face, direction = move
        penalty = 0
        
        # Check for exact repeats
        for i, prev_move in enumerate(reversed(move_history)):
            prev_face, prev_direction = prev_move
            # Penalty for repeating the most recent moves is higher
            if face == prev_face and direction == prev_direction:
                penalty += 50 / (i + 1)
        
        # Higher penalty for move that undoes the previous move
        if len(move_history) > 0:
            last_face, last_direction = move_history[-1]
            if face == last_face:
                if (direction == "clockwise" and last_direction == "counterclockwise") or \
                   (direction == "counterclockwise" and last_direction == "clockwise") or \
                   (direction == "double" and last_direction == "double"):
                    penalty += 100
        
        # Check for repeating patterns (like R U R' U')
        if len(move_history) >= 4:
            # Check if we're repeating a 2-move pattern
            if move == move_history[-2] and move_history[-1] == move_history[-3]:
                penalty += 75
                
            # Check if we're repeating a 4-move pattern
            if len(move_history) >= 8:
                last_four = move_history[-4:]
                prev_four = move_history[-8:-4]
                if last_four == prev_four:
                    penalty += 100
        
        return penalty

    def _detect_loop(self, move_history):
        """
        Detect if we're stuck in a loop of repeated moves
        
        Args:
            move_history: List of previous moves
            
        Returns:
            Boolean indicating if a loop is detected
        """
        if len(move_history) < 6:
            return False
        
        # Check for 2-move loop like (R U R U R U...)
        if len(move_history) >= 6:
            if move_history[-1] == move_history[-3] == move_history[-5] and \
               move_history[-2] == move_history[-4] == move_history[-6]:
                return True
        
        # Check for 3-move loop
        if len(move_history) >= 9:
            if move_history[-1] == move_history[-4] == move_history[-7] and \
               move_history[-2] == move_history[-5] == move_history[-8] and \
               move_history[-3] == move_history[-6] == move_history[-9]:
                return True
        
        # Check for 4-move loop
        if len(move_history) >= 12:
            if move_history[-1:-5:-1] == move_history[-5:-9:-1] == move_history[-9:-13:-1]:
                return True
        
        return False

    def _get_inverse_move(self, face, direction):
        """Get the inverse of a move"""
        if direction == "clockwise":
            return (face, "counterclockwise")
        elif direction == "counterclockwise":
            return (face, "clockwise")
        else:  # double
            return (face, "double")  # Double move is its own inverse

    def _advanced_loop_breaker(self, current_state):
        """Apply a sequence of moves designed to escape a loop while preserving progress"""
        # Get the current progress status
        cross_score = self._evaluate_white_cross(current_state)
        
        # Choose breaker based on current progress
        if cross_score >= 75:  # White cross nearly complete
            # Use gentler algorithms that don't destroy the cross
            breakers = [
                [("U", "clockwise"), ("U", "clockwise")],
                [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise")],
                [("F", "clockwise"), ("U", "clockwise"), ("F", "counterclockwise")]
            ]
        else:
            # More disruptive breakers for early stages
            breakers = [
                [("U", "double"), ("R", "double"), ("F", "double")],
                [("L", "clockwise"), ("U", "double"), ("L", "counterclockwise")],
                [("R", "clockwise"), ("D", "clockwise"), ("R", "counterclockwise")]
            ]
        
        breaker = random.choice(breakers)
        return breaker

    def solve(self, cube_state, controller=None):
        """Solve the cube using reinforcement learning with real-time visualization"""
        print("Attempting to solve cube...")
        
        # Check for None input
        if cube_state is None:
            print("ERROR: Cannot solve a None cube state")
            return []
        
        try:
            # Deep copy to avoid modifying the original
            current_state = copy.deepcopy(cube_state)
            solution_moves = []
            
            # Initialize best states
            best_score_overall = self._evaluate_state(current_state)
            best_state_overall = copy.deepcopy(current_state)
            
            # Initialize white cross tracking
            best_white_cross = self._evaluate_white_cross(current_state)
            best_cross_state = copy.deepcopy(current_state)  # Initialize with current state
            
            # Initialize additional tracking variables
            move_history = []
            history_limit = 10
            steps_without_progress = 0
            local_optima_counter = 0
            visited_states = set()
            exploration_probability = 0.1

            # Add state to visited
            visited_states.add(self._get_state_hash(current_state))
            
            for step in range(self.max_steps):
                print(f"Step {step + 1}/{self.max_steps}")
                
                # Safety check for None state
                if current_state is None:
                    print("ERROR: Current state is None. Using best overall state.")
                    current_state = copy.deepcopy(best_state_overall)
                    # If even that is None, use the original cube state
                    if current_state is None:
                        print("ERROR: Best overall state is also None. Using original state.")
                        current_state = copy.deepcopy(cube_state)
                        # If that fails too, we can't continue
                        if current_state is None:
                            print("FATAL ERROR: All states are None. Cannot continue.")
                            return solution_moves
                
                # Evaluate current state
                current_score = self._evaluate_state(current_state)
                
                # Check for white cross progress
                white_cross_score = self._evaluate_white_cross(current_state)
                if white_cross_score > best_white_cross:
                    best_white_cross = white_cross_score
                    # IMPORTANT: Only update best_cross_state if it's a valid state
                    if current_state is not None:
                        best_cross_state = copy.deepcopy(current_state)
                        print(f"  Improved white cross! Score: {white_cross_score}")
                
                # If we've lost significant white cross progress, revert
                if (best_white_cross > 50 and 
                    white_cross_score < best_white_cross * 0.6 and
                    best_cross_state is not None):
                    print(f"  Lost white cross progress! Reverting to best known cross state")
                    # Revert logic (already implemented)
                    try:
                        current_state = copy.deepcopy(best_cross_state)
                        if current_state is None:
                            raise ValueError("Copied state is None")
                        _ = current_state[0][0][0]
                    except Exception as e:
                        print(f"ERROR when reverting to best cross state: {str(e)}")
                        print("Using best overall state instead")
                        current_state = copy.deepcopy(best_state_overall)
                        if current_state is None:
                            print("ERROR: Best overall state is also None. Using original state.")
                            current_state = copy.deepcopy(cube_state)
                    
                    moves_since_improvement = 0
                    continue
                
                # ACTION SELECTION AND APPLICATION
                # Try each possible action
                action_scores = []
                for action in self.actions:
                    face, direction = action
                    
                    # Apply the action and evaluate the resulting state
                    new_state = self._apply_action(current_state, face, direction)
                    score = self._evaluate_state(new_state)
                    
                    # Calculate penalty for repeating recent moves
                    repeat_penalty = self._calculate_repeat_penalty(move_history, (face, direction))
                    adjusted_score = score - repeat_penalty
                    
                    action_scores.append((action, new_state, score, adjusted_score))
                
                # Sort actions by adjusted score (best first)
                action_scores.sort(key=lambda x: x[3], reverse=True)
                
                # Detect if we're stuck in a loop
                is_in_loop = self._detect_loop(move_history)
                
                # Choose action based on exploration/exploitation strategy
                if is_in_loop:
                    print("  Loop detected! Applying algorithm breaker...")
                    breaker = self._advanced_loop_breaker(current_state)
                    
                    # Apply the breaker algorithm
                    for breaker_face, breaker_dir in breaker:
                        temp_state = self._apply_action(current_state, breaker_face, breaker_dir) 
                        current_state = temp_state
                        solution_moves.append((breaker_face, breaker_dir))
                        move_history.append((breaker_face, breaker_dir))
                        
                        # Update visualization if controller provided
                        if controller:
                            try:
                                controller.model.cube = copy.deepcopy(current_state)
                                controller.view.root.after(0, controller.update_view)
                                import time
                                time.sleep(0.2)
                            except Exception as e:
                                print(f"Error updating visualization: {str(e)}")
                else:
                    # Select action based on exploration probability
                    if random.random() < exploration_probability:
                        # Choose a non-optimal action for exploration
                        if len(action_scores) > 1:
                            chosen_idx = random.randint(1, min(5, len(action_scores)-1))
                            best_action, best_state, best_score, _ = action_scores[chosen_idx]
                            print(f"  Exploring: {best_action}")
                        else:
                            best_action, best_state, best_score, _ = action_scores[0]
                    else:
                        # Choose best action
                        best_action, best_state, best_score, _ = action_scores[0]
                        print(f"  Taking best action: {best_action}")
                    
                    # Apply the selected action
                    face, direction = best_action
                    current_state = best_state
                    solution_moves.append(best_action)
                    move_history.append(best_action)
                    
                    # Trim move history if needed
                    if len(move_history) > history_limit:
                        move_history.pop(0)
                    
                    # Update visualization in real-time if controller is provided
                    if controller:
                        try:
                            controller.model.cube = copy.deepcopy(current_state)
                            controller.view.root.after(0, controller.update_view)
                            import time
                            time.sleep(0.2)
                        except Exception as e:
                            print(f"Error updating visualization: {str(e)}")
                
                # Check if we've reached a solution
                if self._is_solved(current_state):
                    print(f"Solution found in {step + 1} steps!")
                    return solution_moves
                
                # Update best overall state if we've found a better one
                if best_score > best_score_overall:
                    best_score_overall = best_score
                    best_state_overall = copy.deepcopy(current_state)
                    steps_without_progress = 0
                    print(f"  Found better state! Score: {best_score}")
                else:
                    steps_without_progress += 1
                
                # If we haven't made progress for too long, adjust exploration
                if steps_without_progress > 10:
                    exploration_probability = min(0.3, exploration_probability * 1.2)
                    steps_without_progress = 0
                    local_optima_counter += 1
                    print(f"  No progress for several steps, increasing exploration to {exploration_probability:.2f}")
            
            print(f"Maximum steps reached without solution.")
            return solution_moves
        
        except Exception as e:
            print(f"Error in solve method: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def solve_staged(self, cube_state, controller=None):
        """Solve the cube in stages, focusing on one goal at a time"""
        print("Attempting staged solve...")
        
        # Stage 1: White Cross
        print("Stage 1: Solving white cross")
        cross_state = self._solve_white_cross(cube_state, controller)
        
        # Stage 2: White Corners (F2L corners)
        print("Stage 2: Solving white corners")
        corner_state = self._solve_white_corners(cross_state, controller)
        
        # Stage 3: Second layer edges (F2L edges)
        print("Stage 3: Solving second layer")
        f2l_state = self._solve_second_layer(corner_state, controller)
        
        # Stage 4: Yellow face (OLL)
        print("Stage 4: Orienting last layer")
        oll_state = self._solve_yellow_face(f2l_state, controller)
        
        # Stage 5: Final positioning (PLL)
        print("Stage 5: Permuting last layer")
        solution_state = self._solve_final_permutation(oll_state, controller)
        
        return solution_state

    def train(self, num_episodes=1000):
        """
        Train the model using reinforcement learning
        
        Args:
            num_episodes: Number of training episodes
        """
        print(f"Training RL model for {num_episodes} episodes...")
        
        # Create a data model for training
        from model.data_model import DataModel
        model = DataModel()
        
        # Training parameters
        learning_rate = 0.001
        gamma = 0.95  # Discount factor
        epsilon = 1.0  # Exploration rate
        epsilon_min = 0.01
        epsilon_decay = 0.995
        batch_size = 32
        
        # Create memory for replay buffer
        memory = []
        
        # DQN training loop
        for episode in range(num_episodes):
            # Reset the cube to solved state, then scramble it
            model.cube = copy.deepcopy(model.solved)
            self._scramble_cube(model)
            
            current_state = copy.deepcopy(model.cube)
            total_reward = 0
            
            for step in range(self.max_steps):
                # Convert state to features
                state_features = self._state_to_features(current_state)
                
                # Epsilon-greedy action selection
                if np.random.rand() <= epsilon:
                    # Random action
                    action_idx = np.random.randint(0, len(self.actions))
                else:
                    # Best action according to model
                    q_values = self.model.predict(state_features)[0]
                    action_idx = np.argmax(q_values)
                
                # Get the selected action
                action = self.actions[action_idx]
                face, direction = action
                
                # Apply action
                new_state = self._apply_action(current_state, face, direction)
                
                # Calculate reward
                if self._is_solved(new_state):
                    reward = 100  # High reward for solving
                else:
                    # Reward based on improvement in state evaluation
                    prev_score = self._evaluate_state(current_state)
                    new_score = self._evaluate_state(new_state)
                    reward = new_score - prev_score
                
                # Store experience in memory
                memory.append((state_features[0], action_idx, reward, self._state_to_features(new_state)[0], self._is_solved(new_state)))
                
                # Update current state
                current_state = new_state
                total_reward += reward
                
                # If solved, end episode
                if self._is_solved(current_state):
                    print(f"Episode {episode + 1}/{num_episodes}: Solved in {step + 1} steps! Reward: {total_reward}")
                    break
                
                # If we've filled enough memory, perform batch learning
                if len(memory) >= batch_size:
                    # Sample batch
                    batch = random.sample(memory, batch_size)
                    
                    # Extract batch components
                    states = np.array([exp[0] for exp in batch])
                    actions = np.array([exp[1] for exp in batch])
                    rewards = np.array([exp[2] for exp in batch])
                    next_states = np.array([exp[3] for exp in batch])
                    dones = np.array([exp[4] for exp in batch])
                    
                    # Get current Q values
                    current_q = self.model.predict(states)
                    
                    # Get next Q values
                    next_q = self.model.predict(next_states)
                    max_next_q = np.max(next_q, axis=1)
                    
                    # Update Q values for the actions taken
                    targets = current_q.copy()
                    for i in range(batch_size):
                        targets[i, actions[i]] = rewards[i] + (0 if dones[i] else gamma * max_next_q[i])
                    
                    # Train the model
                    self.model.train_on_batch(states, targets)
            
            # Decay epsilon
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            
            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes}: Steps: {step + 1}, Reward: {total_reward}, Epsilon: {epsilon:.4f}")
        
        print("Training completed")

    def _scramble_cube(self, model, moves=20):
        """
        Scramble the cube with random moves
        
        Args:
            model: DataModel instance
            moves: Number of random moves to apply
        """
        faces = ['U', 'D', 'L', 'R', 'F', 'B']
        directions = ['clockwise', 'counterclockwise', 'double']
        
        for _ in range(moves):
            face = random.choice(faces)
            direction = random.choice(directions)
            model.rotate_face(face, direction)
    
    def save_model(self, path):
        """Save the model to the specified path"""
        try:
            self.model.save(path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")


