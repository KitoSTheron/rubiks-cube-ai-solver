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
        self.all_actions = []
        self.stage = "cross"  # Default starting stage
        
        # Populate single moves
        for face in ['U', 'D', 'L', 'R', 'F', 'B']:
            for direction in ['clockwise', 'counterclockwise', 'double']:
                self.all_actions.append({
                    'type': 'single_move',
                    'move_tuple': (face, direction)
                })
        
        # Populate common algorithms
        common_algorithms_list = self._get_common_algorithms()
        for name, moves in common_algorithms_list:
            if moves:  # Ensure algorithm has moves
                self.all_actions.append({
                    'type': 'algorithm',
                    'name': name,
                    'moves_list': moves
                })
        
        # Initialize actions for the starting stage
        self._update_actions_for_stage()
        
        # Set default parameters
        self.max_steps = 100000
        self.default_max_runtime = 300  # 5 minutes in seconds
        
        # Now load the model after actions is defined
        self.model = self._load_or_create_model(model_path)
    
    def _load_or_create_model(self, model_path):
        """Load or create a new TensorFlow model"""
        try:
            pass  # Add your code here
        except Exception as e:
            print(f"An error occurred: {e}")
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
        # self.actions should be populated by __init__ before this is called.
        if not hasattr(self, 'actions') or not self.actions:
            # This is a fallback, should ideally not be needed if __init__ runs first.
            print("Warning: self.actions not defined before _create_model. Defining a basic set.")
            self.actions = []
            for face in ['U', 'D', 'L', 'R', 'F', 'B']:
                for direction in ['clockwise', 'counterclockwise', 'double']:
                     self.actions.append({
                        'type': 'single_move',
                        'move_tuple': (face, direction)
                    })
        
        # Example of a simple DQN network
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(324,)),  # Explicit input layer
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(len(self.actions))  # Q-values for each action/algorithm
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
        # Example of a simple
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

    def _is_f2l_algorithm(self, name):
        """Check if an algorithm is an F2L algorithm."""
        return name.startswith("F2L")

    def _is_oll_algorithm(self, name):
        """Check if an algorithm is an OLL algorithm."""
        return name.startswith("OLL")

    def _is_pll_algorithm(self, name):
        """Check if an algorithm is a PLL algorithm."""
        return name.startswith("PLL")

    def _update_actions_for_stage(self):
        """
        Update self.actions based on the current solving stage.
        """
        if self.stage == "cross":
            # Only single moves
            self.actions = [action for action in self.all_actions if action['type'] == 'single_move']
        elif self.stage == "f2l":
            # Single moves and F2L algorithms in ALL orientations
            self.actions = [
                action for action in self.all_actions
                if action['type'] == 'single_move' or
                (action['type'] == 'algorithm' and (
                    self._is_f2l_algorithm(action['name']) or
                    # Include standard R and L algorithms in all orientations for F2L
                    ("Move" in action['name'] and "oriented" in action['name'].lower())
                ))
            ]
        elif self.stage == "oll":
            # Single moves, OLL algorithms, and U moves
            self.actions = [
                action for action in self.all_actions
                if action['type'] == 'single_move' or
                (action['type'] == 'algorithm' and self._is_oll_algorithm(action['name'])) or
                (action['type'] == 'single_move' and action['move_tuple'][0] == 'U')
            ]
        elif self.stage == "pll":
            # Single moves, PLL algorithms, and U moves
            self.actions = [
                action for action in self.all_actions
                if action['type'] == 'single_move' or
                (action['type'] == 'algorithm' and self._is_pll_algorithm(action['name'])) or
                (action['type'] == 'single_move' and action['move_tuple'][0] == 'U')
            ]
        else:
            # Default to all single moves if stage is unknown
            self.actions = [action for action in self.all_actions if action['type'] == 'single_move']
        
        # Print information about available actions
        alg_count = sum(1 for a in self.actions if a['type'] == 'algorithm')
        move_count = sum(1 for a in self.actions if a['type'] == 'single_move')
        print(f"Stage {self.stage}: {len(self.actions)} actions available ({alg_count} algorithms, {move_count} single moves)")

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
        for move in algorithm:
            face, direction = move[:2]  # Safely unpack only the first two elements
            temp_model.rotate_face(face, direction)
        
        # Return the new state
        return temp_model.cube

    def _get_common_algorithms(self):
        """
        Return a list of common algorithms for Rubik's Cube,
        avoiding M-slice moves and replacing them with R and L based versions
        """
        face_map = {
            "U": ["U", "F", "R", "B", "L"],
            "D": ["D", "F", "R", "B", "L"],
            "F": ["F", "U", "R", "D", "L"],
            "B": ["B", "U", "R", "D", "L"],
            "R": ["R", "U", "F", "D", "B"],
            "L": ["L", "U", "F", "D", "B"]
        }
        
        algorithms = [
            # Basic algorithms
            ("R Move", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise")]),
            ("L Move", [("L", "clockwise"), ("U", "clockwise"), ("L", "counterclockwise"), ("U", "counterclockwise")]),
            ("F Move", [("F", "clockwise"), ("U", "clockwise"), ("F", "counterclockwise"), ("U", "counterclockwise")]),
            ("B Move", [("B", "clockwise"), ("U", "clockwise"), ("B", "counterclockwise"), ("U", "counterclockwise")]),
            ("U Move", [("U", "clockwise"), ("R", "clockwise"), ("U", "counterclockwise"), ("R", "counterclockwise")]),
            ("D Move", [("D", "clockwise"), ("L", "clockwise"), ("D", "counterclockwise"), ("L", "counterclockwise")]),

            # Repeated algorithms (2 to 6 times)
            ("R Move x2", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise")] * 2),
            ("R Move x3", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise")] * 3),
            ("R Move x4", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise")] * 4),
            ("R Move x5", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise")] * 5),
            ("R Move x6", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise")] * 6),

            ("L Move x2", [("L", "clockwise"), ("U", "clockwise"), ("L", "counterclockwise"), ("U", "counterclockwise")] * 2),
            ("L Move x3", [("L", "clockwise"), ("U", "clockwise"), ("L", "counterclockwise"), ("U", "counterclockwise")] * 3),
            ("L Move x4", [("L", "clockwise"), ("U", "clockwise"), ("L", "counterclockwise"), ("U", "counterclockwise")] * 4),
            ("L Move x5", [("L", "clockwise"), ("U", "clockwise"), ("L", "counterclockwise"), ("U", "counterclockwise")] * 5),
            ("L Move x6", [("L", "clockwise"), ("U", "clockwise"), ("L", "counterclockwise"), ("U", "counterclockwise")] * 6),

            ("F Move x2", [("F", "clockwise"), ("U", "clockwise"), ("F", "counterclockwise"), ("U", "counterclockwise")] * 2),
            ("F Move x3", [("F", "clockwise"), ("U", "clockwise"), ("F", "counterclockwise"), ("U", "counterclockwise")] * 3),
            ("F Move x4", [("F", "clockwise"), ("U", "clockwise"), ("F", "counterclockwise"), ("U", "counterclockwise")] * 4),
            ("F Move x5", [("F", "clockwise"), ("U", "clockwise"), ("F", "counterclockwise"), ("U", "counterclockwise")] * 5),
            ("F Move x6", [("F", "clockwise"), ("U", "clockwise"), ("F", "counterclockwise"), ("U", "counterclockwise")] * 6),

            ("B Move x2", [("B", "clockwise"), ("U", "clockwise"), ("B", "counterclockwise"), ("U", "counterclockwise")] * 2),
            ("B Move x3", [("B", "clockwise"), ("U", "clockwise"), ("B", "counterclockwise"), ("U", "counterclockwise")] * 3),
            ("B Move x4", [("B", "clockwise"), ("U", "clockwise"), ("B", "counterclockwise"), ("U", "counterclockwise")] * 4),
            ("B Move x5", [("B", "clockwise"), ("U", "clockwise"), ("B", "counterclockwise"), ("U", "counterclockwise")] * 5),
            ("B Move x6", [("B", "clockwise"), ("U", "clockwise"), ("B", "counterclockwise"), ("U", "counterclockwise")] * 6),

            ("U Move x2", [("U", "clockwise"), ("R", "clockwise"), ("U", "counterclockwise"), ("R", "counterclockwise")] * 2),
            ("U Move x3", [("U", "clockwise"), ("R", "clockwise"), ("U", "counterclockwise"), ("R", "counterclockwise")] * 3),
            ("U Move x4", [("U", "clockwise"), ("R", "clockwise"), ("U", "counterclockwise"), ("R", "counterclockwise")] * 4),
            ("U Move x5", [("U", "clockwise"), ("R", "clockwise"), ("U", "counterclockwise"), ("R", "counterclockwise")] * 5),
            ("U Move x6", [("U", "clockwise"), ("R", "clockwise"), ("U", "counterclockwise"), ("R", "counterclockwise")] * 6),

            ("D Move x2", [("D", "clockwise"), ("L", "clockwise"), ("D", "counterclockwise"), ("L", "counterclockwise")] * 2),
            ("D Move x3", [("D", "clockwise"), ("L", "clockwise"), ("D", "counterclockwise"), ("L", "counterclockwise")] * 3),
            ("D Move x4", [("D", "clockwise"), ("L", "clockwise"), ("D", "counterclockwise"), ("L", "counterclockwise")] * 4),
            ("D Move x5", [("D", "clockwise"), ("L", "clockwise"), ("D", "counterclockwise"), ("L", "counterclockwise")] * 5),
            ("D Move x6", [("D", "clockwise"), ("L", "clockwise"), ("D", "counterclockwise"), ("L", "counterclockwise")] * 6),

            # OLL Algorithms (Orient Last Layer)
            ("OLL - Dot Case", [("F", "clockwise"), ("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise"), ("F", "counterclockwise")]),
            ("OLL - T Shape", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise"), ("R", "counterclockwise"), ("F", "clockwise"), ("R", "F", "counterclockwise")]),
            ("OLL - L Shape", [("F", "clockwise"), ("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise"), ("F", "counterclockwise")]),
            ("OLL - Line", [("F", "clockwise"), ("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise"), ("F", "counterclockwise")]),
            ("OLL - Cross", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "clockwise"), ("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise")]),

            # PLL Algorithms (Permute Last Layer) - Replace M-slice algorithms with R and L based versions
            ("PLL - U Perm", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "clockwise"), ("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise")]),
            
            # Replace H-Perm (originally uses M slice)
            ("PLL - H Perm Alt", [
                ("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"),
                ("U", "clockwise"), ("R", "clockwise"), ("U", "clockwise"), ("U", "clockwise"),
                ("R", "counterclockwise"), ("U", "clockwise")
            ]),
            
            # Replace Z-Perm (originally uses M slice)
            ("PLL - Z Perm Alt", [
                ("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), 
                ("U", "clockwise"), ("R", "clockwise"), ("U", "clockwise"), 
                ("R", "counterclockwise"), ("U", "counterclockwise"), 
                ("R", "counterclockwise"), ("U", "counterclockwise")
            ]),
            
            ("PLL - A Perm", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("F", "clockwise"), ("R", "clockwise"), ("U", "counterclockwise"), ("R", "counterclockwise"), ("F", "counterclockwise")]),
            ("PLL - E Perm", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise"), ("R", "clockwise"), ("F", "clockwise"), ("R", "counterclockwise"), ("F", "counterclockwise")]),

            # F2L Algorithms (First Two Layers)
            ("F2L - Pair Insert", [("U", "clockwise"), ("R", "clockwise"), ("U", "counterclockwise"), ("R", "counterclockwise"), ("U", "counterclockwise"), ("F", "clockwise"), ("U", "clockwise"), ("F", "counterclockwise")]),
            ("F2L - Edge Flip", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "clockwise"), ("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise")]),
            ("F2L - Corner Insert", [("U", "clockwise"), ("R", "clockwise"), ("U", "counterclockwise"), ("R", "counterclockwise"), ("U", "clockwise"), ("F", "clockwise"), ("U", "counterclockwise"), ("F", "counterclockwise")]),

            # Additional algorithms for advanced solving
            ("OLL - Fish Shape", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "clockwise"), ("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise")]),
            ("PLL - T Perm", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise"), ("R", "counterclockwise"), ("F", "clockwise"), ("R", "clockwise"), ("F", "counterclockwise")]),
            ("PLL - J Perm", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("F", "clockwise"), ("R", "clockwise"), ("U", "counterclockwise"), ("R", "counterclockwise"), ("F", "counterclockwise")]),
        ]

        # Make sure we don't have any algorithms with M moves
        supported_faces = ["U", "D", "L", "R", "F", "B"]
        filtered_algorithms = []
        
        for name, moves in algorithms:
            valid = True
            for move in moves:
                face = move[0]
                if face not in supported_faces:
                    valid = False
                    break
            if valid:
                filtered_algorithms.append((name, moves))
        
        standard_faces = ["U", "F", "R", "B", "L", "D"]
        all_oriented_algorithms = []
        
        for name, moves in filtered_algorithms:
            # Add the original algorithm without orientation changes
            all_oriented_algorithms.append((name, moves))
            
            # Create oriented versions
            for orientation in face_map.keys():
                oriented_moves = []
                for move in moves:
                    face, direction = move[:2]  # Safely unpack only the first two elements
                    # Check if face is in our standard faces list
                    if face in standard_faces:
                        # Handle D face specially since it's not in the mapping arrays
                        if face == "D":
                            # For D face, we need a special mapping
                            if orientation == "U": mapped_face = "D"
                            elif orientation == "D": mapped_face = "U"
                            elif orientation == "F": mapped_face = "B"
                            elif orientation == "B": mapped_face = "F"
                            elif orientation == "R": mapped_face = "L"
                            elif orientation == "L": mapped_face = "R"
                        else:
                            # For all other faces, use the mapping
                            face_idx = ["U", "F", "R", "B", "L"].index(face)
                            mapped_face = face_map[orientation][face_idx]
                        
                        oriented_moves.append((mapped_face, direction))
                    else:
                        # For faces not recognized, keep them unchanged
                        oriented_moves.append((face, direction))
                
                # Add the oriented algorithm
                all_oriented_algorithms.append((f"{name} (oriented {orientation})", oriented_moves))

        return all_oriented_algorithms

    def _evaluate_state(self, cube_state):
        """
        Comprehensive evaluation of cube state with progressive scoring
        
        This evaluates multiple solving stages in sequence:
        - White cross
        - White corners
        - F2L (First Two Layers)
        - OLL (Orient Last Layer)
        - PLL (Permute Last Layer)
        """
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
        if cube_state[bottom_face][1][1][0] == 'w':
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
            
            
            
            # --- PROGRESSIVE STAGE EVALUATION ---
            # Bonus for complete white cross
            if white_edges_correct == 4:
                score += 50  # Higher bonus for completed cross
                # --- WHITE CORNERS EVALUATION ---
                corners_score = self._evaluate_white_corners(cube_state)
                score += corners_score  # Add the corners score to the total
                
                # Only proceed to F2L evaluation if corners are completely solved
                white_corners_complete = corners_score >= 150  # The value when all corners are correct
                
                if white_corners_complete:
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
                        # Check if the top face is all oriented correctly (all yellow)
                        top_face_colors = [sticker[0] for row in cube_state[top_face] for sticker in row]
                        if all(color == 'y' for color in top_face_colors):
                            score += 150  # Bonus for completing OLL
                            
                            # Evaluate PLL (Permute Last Layer)
                            if self._is_solved(cube_state):
                                score += 200  # Bonus for solving the cube
            
        return score

    def _evaluate_f2l(self, cube_state):
        """
        Evaluate the F2L (First Two Layers) stage.
        Ensures the correct edge pieces are in the middle layer.
        Only scores if the corners are fully solved.
        """
        if self._evaluate_white_corners(cube_state) < 100:
            # Corners must be fully solved to score F2L
            return 0

        score = 0
        front_face = 1  # Blue face
        right_face = 2  # Red face
        left_face = 3   # Orange face
        back_face = 4   # Green face

        # Define the expected edge positions and their colors
        f2l_edges_positions = [
            (front_face, (1, 2), right_face, (1, 0)),  # Front-right edge
            (front_face, (1, 0), left_face, (1, 2)),   # Front-left edge
            (back_face, (1, 2), left_face, (1, 0)),     # Back-left edge
            (back_face, (1, 0), right_face, (1, 2))    # Back-right edge
        ]
        expected_colors = [
            ['b', 'r'],  # Front-right
            ['b', 'o'],  # Front-left
            ['g', 'o'],  # Back-left
            ['g', 'r']   # Back-right
        ]

        for i, (face1, pos1, face2, pos2) in enumerate(f2l_edges_positions):
            expected_color1, expected_color2 = expected_colors[i]
            if cube_state[face1][pos1[0]][pos1[1]][0] == expected_color1 and \
            cube_state[face2][pos2[0]][pos2[1]][0] == expected_color2:
                score += 25  # Score for each correct edge

        # Bonus for completing all F2L edges
        if score == 100:
            score += 50  # Bonus for completing F2L

        return score
    
    def _evaluate_oll(self, cube_state):
        """
        Evaluate the OLL (Orient Last Layer) stage.
        Ensures all stickers on the top face are yellow.
        Only scores if F2L is fully complete.
        """
        if self._evaluate_f2l(cube_state) < 100:
            # F2L must be fully solved to score OLL
            return 0

        score = 0
        top_face = 0  # Yellow face

        # Check if all stickers on the top face are yellow
        top_face_colors = [sticker[0] for row in cube_state[top_face] for sticker in row]
        if all(color == 'y' for color in top_face_colors):
            score += 100  # Full OLL completion bonus

        return score
    

    def _evaluate_pll(self, cube_state):
        """
        Evaluate the PLL (Permute Last Layer) stage.
        Ensures all pieces on the top face are in their correct positions.
        Only scores if OLL is fully complete.
        """
        if self._evaluate_oll(cube_state) < 100:
            # OLL must be fully solved to score PLL
            return 0

        score = 0
        top_face = 0  # Yellow face
        front_face = 1  # Blue face
        right_face = 2  # Red face
        left_face = 3   # Orange face
        back_face = 4   # Green face

        # Define the expected colors for the top layer edges
        expected_top_edges = [
            (front_face, (0, 1), 'b'),  # Front edge
            (right_face, (0, 1), 'r'),  # Right edge
            (back_face, (0, 1), 'g'),   # Back edge
            (left_face, (0, 1), 'o')    # Left edge
        ]

        # Check if all top edges are in the correct positions
        for face, pos, expected_color in expected_top_edges:
            if cube_state[face][pos[0]][pos[1]][0] == expected_color:
                score += 25  # Score for each correct edge

        # Define the expected colors for the top layer corners
        expected_top_corners = [
            (front_face, (0, 0), right_face, (0, 2), 'b', 'r'),  # Front-right corner
            (front_face, (0, 2), left_face, (0, 0), 'b', 'o'),   # Front-left corner
            (back_face, (0, 2), right_face, (0, 0), 'g', 'r'),   # Back-right corner
            (back_face, (0, 0), left_face, (0, 2), 'g', 'o')     # Back-left corner
        ]

        for face1, pos1, face2, pos2, expected_color1, expected_color2 in expected_top_corners:
            if cube_state[face1][pos1[0]][pos1[1]][0] == expected_color1 and \
            cube_state[face2][pos2[0]][pos2[1]][0] == expected_color2:
                score += 25  # Score for each correct corner

        # Bonus for completing PLL
        if score == 200:
            score += 100  # Bonus for completing PLL

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

    def _evaluate_white_corners(self, cube_state):
        """
        Evaluate white corners to ensure the correct corner pieces are placed 
        in their correct positions and orientations. Only rewards score when 
        the exact correct corner piece is in place AND the cross is complete.
        """
        if cube_state is None:
            print("WARNING: Received None cube state in _evaluate_white_corners")
            return 0
            
        score = 0
        bottom_face = 5  # White face (D)
        front_face = 1   # Blue face (F)
        right_face = 2   # Red face (R)
        left_face = 3    # Orange face (L)
        back_face = 4    # Green face (B)
        
        # First check if the cross is complete
        cross_complete = True
        white_edge_positions = [(0, 1), (1, 0), (1, 2), (2, 1)]
        adjacent_faces = [front_face, left_face, right_face, back_face]
        adjacent_pos = [(2, 1), (2, 1), (2, 1), (2, 1)]
        expected_colors = ['b', 'o', 'r', 'g']
        
        for i, pos in enumerate(white_edge_positions):
            row, col = pos
            adj_row, adj_col = adjacent_pos[i]
            adj_face = adjacent_faces[i]
            
            if not (cube_state[bottom_face][row][col][0] == 'w' and 
                    cube_state[adj_face][adj_row][adj_col][0] == expected_colors[i]):
                cross_complete = False
                break
        
        # Only evaluate corners if cross is complete and white center is correct
        if cross_complete and cube_state[bottom_face][1][1][0] == 'w':
            # Define corner positions on the white face
            corner_positions = [
                (0, 0),  # Front-left
                (0, 2),  # Front-right
                (2, 0),  # Back-left
                (2, 2)   # Back-right
            ]
            
            # Define expected sticker colors for each corner position
            # Corner colors are arranged as [white face color, adjacent face 1 color, adjacent face 2 color]
            expected_corner_colors = [
                ['w', 'b', 'o'],  # Front-left: white, blue, orange
                ['w', 'b', 'r'],  # Front-right: white, blue, red
                ['w', 'g', 'o'],  # Back-left: white, green, orange
                ['w', 'g', 'r']   # Back-right: white, green, red
            ]
            
            # Adjacent face positions for each corner
            adjacent_faces = [
                [(front_face, (2, 0)), (left_face, (2, 2))],    # Front-left
                [(front_face, (2, 2)), (right_face, (2, 0))],   # Front-right
                [(back_face, (2, 2)), (left_face, (2, 0))],     # Back-left
                [(back_face, (2, 0)), (right_face, (2, 2))]     # Back-right
            ]
            
            corners_correct = 0
            
            # Check each corner
            for i, pos in enumerate(corner_positions):
                row, col = pos
                
                # Get the actual colors of this corner piece from all three faces
                actual_colors = [
                    cube_state[bottom_face][row][col][0],  # White face sticker
                    cube_state[adjacent_faces[i][0][0]][adjacent_faces[i][0][1][0]][adjacent_faces[i][0][1][1]][0],  # First adjacent face
                    cube_state[adjacent_faces[i][1][0]][adjacent_faces[i][1][1][0]][adjacent_faces[i][1][1][1]][0]   # Second adjacent face
                ]
                
                # Check if we have the right piece in the right position (by checking all colors match)
                if sorted(actual_colors) == sorted(expected_corner_colors[i]):
                    score += 20  # Reward for having the right piece in the right position
                    
                    # Additional reward for correct orientation (white on bottom)
                    if actual_colors[0] == 'w':
                        score += 10
                        
                        # Perfect position and orientation
                        if actual_colors == expected_corner_colors[i]:
                            score += 20  # Extra reward for perfect orientation
                            corners_correct += 1
            
            # Massive bonus for all corners correct
            if corners_correct == 4:
                score += 50
                    
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

    def _get_inverse_move(self, face, direction):
        """Get the inverse of a move"""
        if direction == "clockwise":
            return (face, "counterclockwise")
        elif direction == "counterclockwise":
            return (face, "clockwise")
        else:  # double
            return (face, "double")  # Double move is its own inverse


    def solve(self, cube_state, controller=None, max_runtime=None):
        """
        Solve the cube using reinforcement learning with real-time visualization

        Args:
            cube_state: The current cube state to solve
            controller: Optional controller for visualization updates
            max_runtime: Maximum runtime in seconds (default: 300s/5min)

        Returns:
            List of solution moves
        """
        print("Attempting to solve cube...")

        import time
        import numpy as np  # For softmax
        start_time = time.time()
        if max_runtime is None:
            max_runtime = self.default_max_runtime

        if cube_state is None:
            print("ERROR: Cannot solve a None cube state")
            return []

        try:
            current_state = copy.deepcopy(cube_state)
            solution_moves = []

            best_score_overall = self._evaluate_state(current_state)
            best_state_overall = copy.deepcopy(current_state)

            best_white_cross = self._evaluate_white_cross(current_state)
            best_cross_state = copy.deepcopy(current_state)
            best_cross_step = 0  # Track when best cross was achieved

            best_corners_score = 0
            best_corners_state = None
            best_corners_step = 0
            corners_regress_counter = 0
            corners_regress_limit = 6  # More strict than cross regression

            move_history = []
            visited_states = set()
            exploration_probability = 0.15  # Base exploration rate

            white_cross_solved = False
            f2l_solved = False
            oll_solved = False
            pll_solved = False

            initial_hash = self._get_state_hash(current_state)
            visited_states.add(initial_hash)

            cross_regress_counter = 0
            cross_regress_limit = 8  # Allow up to 8 moves of cross regression before reverting

            # RL experience replay buffer
            experience_buffer = []
            batch_size = 32
            gamma = 0.95  # Discount factor

            for step in range(self.max_steps):
                # Update the stage based on progress
                if not white_cross_solved and self._evaluate_white_cross(current_state) >= 100:
                    white_cross_solved = True
                    # Don't change the stage yet - we still need to solve corners
                    print("White cross solved! Now solving corners...")
                elif white_cross_solved and self._evaluate_white_corners(current_state) >= 150 and self.stage == "cross":
                    # Now move to the F2L stage after corners are solved
                    self.stage = "f2l"
                    self._update_actions_for_stage()
                    print("White corners solved! Moving to F2L...")
                elif white_cross_solved and not f2l_solved and self._evaluate_f2l(current_state) >= 100:
                    f2l_solved = True
                    self.stage = "oll"
                    self._update_actions_for_stage()
                    print("F2L solved! Moving to OLL...")
                elif f2l_solved and not oll_solved and self._evaluate_oll(current_state) >= 100:
                    oll_solved = True
                    self.stage = "pll"
                    self._update_actions_for_stage()
                    print("OLL solved! Moving to PLL...")
                elif oll_solved and not pll_solved and self._evaluate_pll(current_state) >= 100:
                    pll_solved = True
                    print("PLL solved! Cube is solved!")

                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time > max_runtime:
                    print(f"Time limit of {max_runtime} seconds reached. Stopping.")
                    return solution_moves

                print(f"Step {step + 1}/{self.max_steps} (Elapsed time: {elapsed_time:.1f}s)")

                if current_state is None:
                    print("ERROR: Current state is None. Using best overall state.")
                    current_state = copy.deepcopy(best_state_overall)
                    if current_state is None: 
                        print("ERROR: Best overall state is also None. Cannot continue.")
                        return solution_moves
                    visited_states.add(self._get_state_hash(current_state)) 

                current_score_eval = self._evaluate_state(current_state) 
                white_cross_score = self._evaluate_white_cross(current_state)

                # Update solved status
                if white_cross_score >= 100 and not white_cross_solved:
                    white_cross_solved = True
                    print("White cross completed! Continuing with solving...")

                # Update best scores and states for stages
                if white_cross_score > best_white_cross:
                    best_white_cross = white_cross_score
                    best_cross_state = copy.deepcopy(current_state)
                    best_cross_step = step
                    cross_regress_counter = 0
                    print(f"  Improved white cross! Score: {white_cross_score:.0f}")

                corners_score = self._evaluate_white_corners(current_state)
                if corners_score > best_corners_score:
                    best_corners_score = corners_score
                    best_corners_state = copy.deepcopy(current_state)
                    best_corners_step = step
                    corners_regress_counter = 0
                    print(f"  Improved white corners! Score: {corners_score:.0f}")

                # Allow temporary regression, but revert if cross progress is lost for too long
                if not white_cross_solved:
                    if white_cross_score < best_white_cross * 0.7:
                        cross_regress_counter += 1
                        print(f"  Cross regression detected ({cross_regress_counter}/{cross_regress_limit})")
                        if cross_regress_counter >= cross_regress_limit:
                            print("  Too much cross regression, reverting to best cross state.")
                            current_state = copy.deepcopy(best_cross_state)
                            # Remove moves since best_cross_step
                            solution_moves = solution_moves[:best_cross_step]
                            move_history = move_history[:best_cross_step]
                            cross_regress_counter = 0
                            continue
                    else:
                        cross_regress_counter = 0

                # Handle corner regression
                if white_cross_solved and not f2l_solved:
                    if corners_score < best_corners_score * 0.8:  # Less tolerance for corner regression
                        corners_regress_counter += 1
                        print(f"  Corner regression detected ({corners_regress_counter}/{corners_regress_limit})")
                        if corners_regress_counter >= corners_regress_limit:
                            print("  Too much corner regression, reverting to best corners state.")
                            current_state = copy.deepcopy(best_corners_state)
                            solution_moves = solution_moves[:best_corners_step]
                            move_history = move_history[:best_corners_step]
                            corners_regress_counter = 0
                            continue
                    else:
                        corners_regress_counter = 0

                all_evaluated_options = []

                # --- RL: Predict Q-values for all actions ---
                state_features = self._state_to_features(current_state)
                q_values = None
                if hasattr(self, "model") and self.model is not None:
                    try:
                        q_values = self.model.predict(state_features, verbose=0)[0]
                    except Exception as e:
                        print(f"Warning: Model prediction failed: {e}")
                        q_values = None

                for idx, action_definition in enumerate(self.actions):
                    temp_eval_state = copy.deepcopy(current_state)
                    action_type = action_definition['type']

                    option_entry = {"type": action_type}
                    first_move_for_penalty_calc = None

                    if action_type == 'single_move':
                        face, direction = action_definition['move_tuple']
                        temp_eval_state = self._apply_action(temp_eval_state, face, direction)
                        option_entry["move_tuple"] = action_definition['move_tuple']
                        first_move_for_penalty_calc = action_definition['move_tuple']
                    elif action_type == 'algorithm':
                        alg_name = action_definition['name']
                        alg_moves = action_definition['moves_list']
                        if not alg_moves: continue
                        temp_eval_state = self._apply_algorithm(temp_eval_state, alg_moves)
                        option_entry["name"] = alg_name
                        option_entry["moves_list"] = alg_moves
                        first_move_for_penalty_calc = alg_moves[0] if alg_moves else None
                    else:
                        print(f"Warning: Unknown action type: {action_type}")
                        continue

                    original_score = self._evaluate_state(temp_eval_state)

                    repeat_penalty = 0
                    if first_move_for_penalty_calc:
                        repeat_penalty = self._calculate_repeat_penalty(move_history, first_move_for_penalty_calc)
                        if action_type == 'algorithm':  # Algorithms are more deliberate
                            repeat_penalty *= 0.3 

                    stage_bonus = 0
                    eval_cross_score = self._evaluate_white_cross(temp_eval_state)
                    eval_corners_score = self._evaluate_white_corners(temp_eval_state)

                    # Favor algorithms over single moves, especially for F2L and corners
                    if action_type == 'algorithm':
                        # Strong bonus for using algorithms in appropriate stages
                        if self.stage == "cross" and white_cross_score < 100:
                            stage_bonus += 20  # Small bonus for algorithms during cross
                        elif white_cross_solved and corners_score < 150:
                            stage_bonus += 150  # Big bonus for algorithms during corner solving
                        elif self.stage == "f2l":
                            stage_bonus += 200  # Very big bonus for algorithms during F2L
                        elif self.stage == "oll":
                            stage_bonus += 180  # Big bonus for algorithms during OLL
                        elif self.stage == "pll":
                            stage_bonus += 180  # Big bonus for algorithms during PLL

                    # Penalize excessive U turns when working on F2L and corners
                    if action_type == 'single_move' and action_definition['move_tuple'][0] == 'U':
                        if white_cross_solved and corners_score < 150:
                            stage_bonus -= 40  # Penalty for U turns during corner solving
                        elif self.stage == "f2l":
                            stage_bonus -= 30  # Penalty for U turns during F2L

                    new_state_hash = self._get_state_hash(temp_eval_state)
                    visit_penalty = 200 if new_state_hash in visited_states else 0

                    adjusted_score = original_score + stage_bonus - repeat_penalty - visit_penalty

                    # RL: If Q-values available, blend with adjusted_score
                    if q_values is not None and idx < len(q_values):
                        # Weighted sum: RL Q-value + heuristic
                        rl_weight = 0.6
                        adjusted_score = rl_weight * q_values[idx] + (1 - rl_weight) * adjusted_score

                    option_entry["next_state"] = temp_eval_state
                    option_entry["original_score"] = original_score
                    option_entry["adjusted_score"] = adjusted_score
                    option_entry["action_idx"] = idx
                    all_evaluated_options.append(option_entry)

                all_evaluated_options.sort(key=lambda x: x["adjusted_score"], reverse=True) 

                # --- RL: Epsilon-greedy selection ---
                chosen_option_details = None
                epsilon = exploration_probability

                # Increase exploration during F2L and corners stages
                if white_cross_solved and (not f2l_solved or corners_score < 150):
                    epsilon = max(0.25, epsilon)  # Higher exploration during critical stages

                if q_values is not None:
                    # Anneal epsilon over time
                    epsilon = max(0.05, epsilon * (1 - step / self.max_steps))

                if np.random.rand() < epsilon and len(all_evaluated_options) > 1:
                    # Filter for algorithm options first
                    algorithm_options = [
                        opt for opt in all_evaluated_options 
                        if opt["type"] == "algorithm" and 
                        (white_cross_solved and (not f2l_solved or corners_score < 150))
                    ]
                    
                    # If we have algorithm options and we're in the right stage, prefer those
                    if algorithm_options and white_cross_solved and (not f2l_solved or corners_score < 150):
                        chosen_option_details = np.random.choice(algorithm_options[:min(5, len(algorithm_options))])
                        print(f"  RL Exploring (Algorithm): {chosen_option_details.get('name')}")
                    else:
                        # Otherwise choose from all options
                        chosen_option_details = np.random.choice(all_evaluated_options[1:min(6, len(all_evaluated_options))])
                        print(f"  RL Exploring: {chosen_option_details.get('name', chosen_option_details.get('move_tuple'))}")
                elif all_evaluated_options:
                    chosen_option_details = all_evaluated_options[0]
                    print(f"  RL Exploiting: {chosen_option_details.get('name', chosen_option_details.get('move_tuple'))}")

                # --- RL: Store experience for learning ---
                if chosen_option_details:
                    prev_features = self._state_to_features(current_state)
                    action_idx = chosen_option_details.get("action_idx", 0)
                    reward = chosen_option_details["original_score"] - current_score_eval
                    next_state = chosen_option_details["next_state"]
                    done = self._is_solved(next_state)
                    next_features = self._state_to_features(next_state)
                    experience_buffer.append((prev_features, action_idx, reward, next_features, done))

                    # Keep buffer size reasonable
                    if len(experience_buffer) > 10000:
                        experience_buffer = experience_buffer[-10000:]

                # --- RL: Train model from experience buffer ---
                if hasattr(self, "model") and self.model is not None and len(experience_buffer) >= batch_size:
                    minibatch = random.sample(experience_buffer, batch_size)
                    states = np.vstack([exp[0] for exp in minibatch])
                    actions = [exp[1] for exp in minibatch]
                    rewards = [exp[2] for exp in minibatch]
                    next_states = np.vstack([exp[3] for exp in minibatch])
                    dones = [exp[4] for exp in minibatch]

                    # Predict Q-values for current and next states
                    q_current = self.model.predict(states, verbose=0)
                    q_next = self.model.predict(next_states, verbose=0)

                    for i in range(batch_size):
                        target = rewards[i]
                        if not dones[i]:
                            target += gamma * np.max(q_next[i])
                        q_current[i][actions[i]] = target

                    self.model.fit(states, q_current, epochs=1, verbose=0)

                # --- Apply chosen move ---
                if chosen_option_details:
                    current_state = chosen_option_details["next_state"]
                    visited_states.add(self._get_state_hash(current_state))
                    if chosen_option_details["type"] == "single_move":
                        solution_moves.append(chosen_option_details["move_tuple"])
                        move_history.append(chosen_option_details["move_tuple"])
                    elif chosen_option_details["type"] == "algorithm":
                        solution_moves.extend(chosen_option_details["moves_list"])
                        move_history.extend(chosen_option_details["moves_list"])

                    if controller and hasattr(controller.view, 'root'):
                        controller.model.cube = copy.deepcopy(current_state)
                        controller.view.root.after(0, lambda: controller.view.update_view(controller.model.cube))
                        controller.view.root.update_idletasks()
                        controller.view.root.update()
                        time.sleep(0.01)

                if self._is_solved(current_state):
                    print(f"Solution found in {step + 1} steps!")
                    return solution_moves

            print(f"Maximum steps reached. Current solution length: {len(solution_moves)}")
            return solution_moves

        except Exception as e:
            print(f"Error in solve method: {str(e)}")
            import traceback
            traceback.print_exc()
            return []