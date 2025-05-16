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
        
        # Set default parameters
        self.max_steps = 100000  # Increased from 100 to 1000
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
        Return a list of common algorithms for Rubik's Cube,
        avoiding M-slice moves and replacing them with R and L moves
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

    def _get_solved_corners(self, cube_state):
        """Return a set of indices of correctly placed and oriented corners"""
        solved_corners = set()
        bottom_face = 5  # White face
        front_face = 1   # Blue face
        right_face = 2   # Red face
        left_face = 3    # Orange face
        back_face = 4    # Green face
        
        if cube_state[bottom_face][1][1][0] != 'w':
            return solved_corners
        
        corner_positions = [(0, 0), (0, 2), (2, 0), (2, 2)]
        expected_colors = [
            ['w', 'b', 'o'],  # Front-left: white, blue, orange
            ['w', 'b', 'r'],  # Front-right: white, blue, red
            ['w', 'g', 'o'],  # Back-left: white, green, orange
            ['w', 'g', 'r']   # Back-right: white, green, red
        ]
        
        adjacent_faces = [
            [(front_face, (2, 0)), (left_face, (2, 2))],
            [(front_face, (2, 2)), (right_face, (2, 0))],
            [(back_face, (2, 2)), (left_face, (2, 0))],
            [(back_face, (2, 0)), (right_face, (2, 2))]
        ]
        
        for i, pos in enumerate(corner_positions):
            row, col = pos
            actual_colors = [
                cube_state[bottom_face][row][col][0],
                cube_state[adjacent_faces[i][0][0]][adjacent_faces[i][0][1][0]][adjacent_faces[i][0][1][1]][0],
                cube_state[adjacent_faces[i][1][0]][adjacent_faces[i][1][1][0]][adjacent_faces[i][1][1][1]][0]
            ]
            
            if actual_colors == expected_colors[i]:
                solved_corners.add(i)
        
        return solved_corners

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
        Enhanced detection for identifying more loop patterns
        """
        if len(move_history) < 6:
            return False # Not enough move history for subsequent checks if state checks didn't trigger

        # Check for 2-move loop like (R U R U R U...)
        if move_history[-1] == move_history[-3] == move_history[-5] and \
           move_history[-2] == move_history[-4] == move_history[-6]:
            print("  Move-based loop detected (A B A B A B).")
            return True
        
        # Check for 3-move loop
        if len(move_history) >= 9:
            if move_history[-1] == move_history[-4] == move_history[-7] and \
               move_history[-2] == move_history[-5] == move_history[-8] and \
               move_history[-3] == move_history[-6] == move_history[-9]:
                print("  Move-based loop detected (A B C A B C A B C).")
                return True
        
        # Check for 4-move loop
        if len(move_history) >= 12:
            pattern_4 = move_history[-4:]
            if move_history[-8:-4] == pattern_4 and move_history[-12:-8] == pattern_4:
                 print("  Move-based loop detected (A B C D pattern x3).")
                 return True
        
        # Check for longer patterns that might be missed
        if len(move_history) >= 16:
            for pattern_length in range(2, 8):
                if len(move_history) >= pattern_length * 3:
                    recent_pattern = move_history[-pattern_length:]
                    is_repeating = True
                    for i in range(1, 3):
                        offset = pattern_length * i
                        # Ensure indices are valid for comparison
                        start_compare_idx = -pattern_length - offset
                        end_compare_idx = -offset if -offset != 0 else None
                        
                        if len(move_history) < pattern_length + offset : # Not enough history to compare
                            is_repeating = False
                            break

                        compare_pattern = move_history[start_compare_idx : end_compare_idx]
                        if recent_pattern != compare_pattern:
                            is_repeating = False
                            break
                    if is_repeating:
                        print(f"  Move-based loop detected (longer pattern, length {pattern_length} x3).")
                        return True
        
        # Detect oscillation (alternating between two small sets of moves)
        if len(move_history) >= 8:
            set1 = frozenset(move_history[-4:])
            set2 = frozenset(move_history[-8:-4])
            if len(set1) <= 2 and len(set2) <= 2 and set1 == set2:
                print("  Move-based oscillation detected (alternating small sets of moves).")
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
        
        # Use a random breaker regardless of progress
        breakers = [
            [("U", "clockwise"), ("U", "clockwise")],
            [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise")],
            [("F", "clockwise"), ("U", "clockwise"), ("F", "counterclockwise")],
            [("L", "clockwise"), ("U", "clockwise"), ("L", "counterclockwise")],
            [("U", "double"), ("R", "double"), ("F", "double")],
            [("L", "clockwise"), ("U", "double"), ("L", "counterclockwise")],
            [("R", "clockwise"), ("D", "clockwise"), ("R", "counterclockwise")],
            [("R", "clockwise"), ("F", "clockwise"), ("R", "counterclockwise"), ("F", "counterclockwise")],
            [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "clockwise"), ("R", "clockwise"), ("U", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise")],
        ]
        
        # Apply different orientations to the breaker algorithms
        face_map = {
            "U": ["U", "F", "R", "B", "L"],
            "D": ["D", "F", "R", "B", "L"],
            "F": ["F", "U", "R", "D", "L"],
            "B": ["B", "U", "R", "D", "L"],
            "R": ["R", "U", "F", "D", "B"],
            "L": ["L", "U", "F", "D", "B"]
        }
        
        # Choose a random breaker first
        selected_breaker = random.choice(breakers)
        
        # Choose a random orientation too
        orientation = random.choice(list(face_map.keys()))
        
        # Apply the orientation mapping to the selected breaker
        oriented_breaker = []
        for face, direction in selected_breaker:
            mapping_list = ["U", "F", "R", "B", "L"]
            if face in mapping_list:
                mapped_face = face_map[orientation][mapping_list.index(face)]
                oriented_breaker.append((mapped_face, direction))
            else:
                oriented_breaker.append((face, direction))

        return oriented_breaker

    def _get_next_target_after_cross(self, cube_state):
        """
        When cross is complete, determine what to focus on next
        Returns a focus area and corresponding algorithm if applicable
        """
        bottom_face = 5  # White face
        cross_score = self._evaluate_white_cross(cube_state)
        corners_score = self._evaluate_white_corners(cube_state)
        
        # If cross is solved but corners aren't, focus on corners
        if cross_score >= 90 and corners_score < 90:
            # Check which corners need solving
            corner_positions = [(0, 0), (0, 2), (2, 0), (2, 2)]
            front_face = 1
            right_face = 2
            left_face = 3
            back_face = 4
            
            # Adjacent face indices for each corner
            adjacent_faces = [
                [(front_face, (2, 0)), (left_face, (2, 2))],    # Front-left
                [(front_face, (2, 2)), (right_face, (2, 0))],   # Front-right
                [(back_face, (2, 2)), (left_face, (2, 0))],     # Back-left
                [(back_face, (2, 0)), (right_face, (2, 2))]     # Back-right
            ]
            
            for i, pos in enumerate(corner_positions):
                row, col = pos
                
                # Skip if this corner is already correct
                if cube_state[bottom_face][row][col][0] == 'w' and \
                   cube_state[adjacent_faces[i][0][0]][adjacent_faces[i][0][1][0]][adjacent_faces[i][0][1][1]][0] == cube_state[adjacent_faces[i][0][0]][1][1][0] and \
                   cube_state[adjacent_faces[i][1][0]][adjacent_faces[i][1][1][0]][adjacent_faces[i][1][1][1]][0] == cube_state[adjacent_faces[i][1][0]][1][1][0]:
                    continue
                
                # Find a relevant algorithm for this corner
                return "corner", i, None
        
        # If corners are also done, focus on the middle layer (F2L edges)
        elif cross_score >= 90 and corners_score >= 90:
            return "f2l", None, None
            
        # Default: keep working on cross
        return "cross", None, None

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
        import numpy as np # For softmax
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
            
            best_white_corners = self._evaluate_white_corners(current_state)
            best_corners_state = copy.deepcopy(current_state)
            
            move_history = []
            state_hash_history = [] # Initialize state_hash_history
            state_hash_history_limit = 20 # Define limit for state_hash_history

            history_limit = 10 
            steps_without_progress = 0
            local_optima_counter = 0
            visited_states = set()
            exploration_probability = 0.1 
            
            white_cross_solved = False
            white_corners_solved = False
            
            common_algorithms = self._get_common_algorithms()
            current_stage = "cross"
            current_target = None

            initial_hash = self._get_state_hash(current_state)
            visited_states.add(initial_hash)
            # state_hash_history will be populated at the end of each step
            
            corner_milestone_states = {}
            
            for step in range(self.max_steps):
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
                    visited_states.add(self._get_state_hash(current_state)) # Add to visited if reverted
                
                current_score_eval = self._evaluate_state(current_state) 
                white_cross_score = self._evaluate_white_cross(current_state)
                white_corners_score = self._evaluate_white_corners(current_state)
                
                solved_corners = self._get_solved_corners(current_state)
                num_solved = len(solved_corners)

                if num_solved > 0 and (num_solved not in corner_milestone_states or 
                                        self._evaluate_white_corners(current_state) > self._evaluate_white_corners(corner_milestone_states[num_solved])):
                    corner_milestone_states[num_solved] = copy.deepcopy(current_state)
                    print(f"  New milestone: {num_solved} corners correctly placed!")
                
                best_corners_from_state = self._get_solved_corners(best_corners_state) if best_corners_state is not None else set()
                if len(solved_corners) > len(best_corners_from_state):
                    print(f"  Found new solved corner(s)! Now have {len(solved_corners)} corners solved")
                    best_corners_state = copy.deepcopy(current_state)
                    best_white_corners = white_corners_score 
                
                if white_cross_score >= 100 and not white_cross_solved:
                    white_cross_solved = True
                    print("White cross completed! Moving to next stage...")
                    current_stage, current_target, _ = self._get_next_target_after_cross(current_state)
                    print(f"New focus: {current_stage}")
                
                if white_corners_score >= 100 and not white_corners_solved: 
                    white_corners_solved = True
                    print("White corners completed! Moving to F2L...")
                    current_stage = "f2l"
                
                if white_cross_score > best_white_cross:
                    best_white_cross = white_cross_score
                    best_cross_state = copy.deepcopy(current_state)
                    print(f"  Improved white cross! Score: {white_cross_score}")
                
                if white_cross_solved and white_corners_score > best_white_corners:
                    best_white_corners = white_corners_score
                    best_corners_state = copy.deepcopy(current_state)
                    print(f"  Improved white corners! Score: {white_corners_score}")
                
                # Revert if white cross progress is significantly lost
                if (best_white_cross > 50 and white_cross_score < best_white_cross * 0.6 and best_cross_state is not None):
                    print(f"  Lost white cross progress (current: {white_cross_score:.0f}, best: {best_white_cross:.0f})! Reverting to best known cross state.")
                    current_state = copy.deepcopy(best_cross_state)
                    visited_states.add(self._get_state_hash(current_state))
                    if controller and hasattr(controller.view, 'root'):
                        controller.model.cube = copy.deepcopy(current_state)
                        controller.view.root.after(0, lambda: controller.view.update_view(controller.model.cube))
                        controller.view.root.update_idletasks()
                        time.sleep(0.1)
                    # Update scores after revert for consistency within the step
                    white_cross_score = self._evaluate_white_cross(current_state)
                    white_corners_score = self._evaluate_white_corners(current_state)
                    current_score_eval = self._evaluate_state(current_state)
                    continue

                # Revert if white corner progress is significantly lost (New Logic)
                if (white_cross_solved and 
                    best_corners_state is not None and
                    best_white_corners > 30 and # Ensure best_white_corners is somewhat significant (e.g., at least one corner partially done)
                    white_corners_score < best_white_corners * 0.7): # Revert if current corner score drops significantly from its best
                    print(f"  Lost significant corner progress (current: {white_corners_score:.0f}, best: {best_white_corners:.0f})! Reverting to best known corners state.")
                    current_state = copy.deepcopy(best_corners_state)
                    visited_states.add(self._get_state_hash(current_state))
                    if controller and hasattr(controller.view, 'root'):
                        controller.model.cube = copy.deepcopy(current_state)
                        controller.view.root.after(0, lambda: controller.view.update_view(controller.model.cube))
                        controller.view.root.update_idletasks()
                        time.sleep(0.1)
                    # Update scores after revert
                    white_cross_score = self._evaluate_white_cross(current_state)
                    white_corners_score = self._evaluate_white_corners(current_state)
                    current_score_eval = self._evaluate_state(current_state)
                    continue
                
                evaluated_action_scores = []
                if not hasattr(self, 'actions') or not self.actions:
                    print("ERROR: No actions defined for the solver.")
                    return solution_moves
                
                for action_tuple in self.actions:
                    face, direction = action_tuple
                    new_state_eval = self._apply_action(current_state, face, direction)
                    original_score = self._evaluate_state(new_state_eval)
                    
                    repeat_penalty = self._calculate_repeat_penalty(move_history, action_tuple)
                    stage_bonus = 0
                    
                    new_state_hash = self._get_state_hash(new_state_eval)
                    visit_penalty = 0
                    if new_state_hash in visited_states:
                        visit_penalty = 150 

                    current_solved_for_bonus = self._get_solved_corners(current_state)
                    if len(current_solved_for_bonus) > 0 and white_cross_solved:
                        new_solved_for_bonus = self._get_solved_corners(new_state_eval)
                        if not new_solved_for_bonus.issuperset(current_solved_for_bonus):
                            lost_count = len(current_solved_for_bonus) - len(current_solved_for_bonus.intersection(new_solved_for_bonus))
                            stage_bonus -= 1000 * lost_count
                    
                    if current_stage == "corner" and (face == 'R' or face == 'L'): stage_bonus += 10
                    elif current_stage == "cross" and (face in ['D', 'F', 'B']): stage_bonus += 5
                    elif white_cross_score >= 90 and face == 'U': stage_bonus += 5
                    
                    if white_cross_score >= 100 and self._evaluate_white_cross(new_state_eval) < 90: stage_bonus -= 1000
                    if white_corners_score >= 100 and self._evaluate_white_corners(new_state_eval) < 90: stage_bonus -= 800
                        
                    adjusted_score = original_score - repeat_penalty + stage_bonus - visit_penalty
                    evaluated_action_scores.append((action_tuple, new_state_eval, original_score, adjusted_score))

                action_applied_by_algorithm = False
                if current_stage == "corner" and white_cross_solved:
                    temp_filtered_scores = []
                    for act_data in evaluated_action_scores:
                        act, n_state, o_score, adj_score = act_data
                        f, d = act
                        if f in ['R', 'L', 'U']:
                            new_adj_score = adj_score
                            if f in ['R', 'L']: new_adj_score += 100
                            else: new_adj_score += 20 
                            temp_filtered_scores.append((act, n_state, o_score, new_adj_score))
                    if temp_filtered_scores:
                        evaluated_action_scores = temp_filtered_scores

                    if random.random() < 0.9: 
                        corner_algs_list = []
                        basic_algs = [("R U R' U'", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise")]), ("L U L' U'", [("L", "clockwise"), ("U", "clockwise"), ("L", "counterclockwise"), ("U", "counterclockwise")]), ("R' U' R U", [("R", "counterclockwise"), ("U", "counterclockwise"), ("R", "clockwise"), ("U", "clockwise")]), ("L' U' L U", [("L", "counterclockwise"), ("U", "counterclockwise"), ("L", "clockwise"), ("U", "clockwise")]), ("U R U' R'", [("U", "clockwise"), ("R", "clockwise"), ("U", "counterclockwise"), ("R", "counterclockwise")]), ("U L U' L'", [("U", "clockwise"), ("L", "clockwise"), ("U", "counterclockwise"), ("L", "counterclockwise")]), ("U' R U R'", [("U", "counterclockwise"), ("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise")]), ("U' L U L'", [("U", "counterclockwise"), ("L", "clockwise"), ("U", "clockwise"), ("L", "counterclockwise")]),]
                        u_moves_list = [("U", [("U", "clockwise")]), ("U'", [("U", "counterclockwise")]), ("U2", [("U", "double")]),]
                        for name, alg_moves in basic_algs:
                            corner_algs_list.append((name, alg_moves))
                            for i in range(1, 7): corner_algs_list.append((f"{name} x{i}", alg_moves * i))
                        corner_algs_list.extend(u_moves_list)
                        
                        if corner_algs_list:
                            chosen_alg_name, chosen_alg_moves = random.choice(corner_algs_list)
                            print(f"  Applying corner algorithm: {chosen_alg_name}")
                            new_state_algo = self._apply_algorithm(copy.deepcopy(current_state), chosen_alg_moves)
                            solved_corners_pre_algo = self._get_solved_corners(current_state)
                            solved_corners_post_algo = self._get_solved_corners(new_state_algo)

                            if len(solved_corners_post_algo) >= len(solved_corners_pre_algo):
                                current_state = new_state_algo
                                visited_states.add(self._get_state_hash(current_state))
                                for move in chosen_alg_moves:
                                    solution_moves.append(move)
                                    move_history.append(move) # Algorithms also contribute to move_history for loop detection
                                if controller and hasattr(controller.view, 'root'):
                                    controller.model.cube = copy.deepcopy(current_state) 
                                    controller.view.root.after(0, lambda: controller.view.update_view(controller.model.cube))
                                    controller.view.root.update_idletasks()
                                    time.sleep(0.05 * len(chosen_alg_moves)) 
                                print(f"  Algorithm applied, now have {len(solved_corners_post_algo)} corners solved!")
                                action_applied_by_algorithm = True
                                continue 
                            else:
                                print(f"  Algorithm would reduce corner progress, trying individual moves...")

                elif current_stage == "f2l" and white_cross_solved and white_corners_solved:
                    if random.random() < 0.5: 
                        f2l_algs_list = [alg for alg in common_algorithms if "F2L" in alg[0]]
                        if f2l_algs_list:
                            chosen_alg_name, chosen_alg_moves = random.choice(f2l_algs_list)
                            print(f"  Applying F2L algorithm: {chosen_alg_name}")
                            new_state_algo = self._apply_algorithm(copy.deepcopy(current_state), chosen_alg_moves)
                            score_post_algo = self._evaluate_state(new_state_algo)
                            if score_post_algo > current_score_eval: 
                                current_state = new_state_algo
                                visited_states.add(self._get_state_hash(current_state))
                                for move in chosen_alg_moves:
                                    solution_moves.append(move)
                                    move_history.append(move) # Algorithms also contribute to move_history
                                if controller and hasattr(controller.view, 'root'):
                                    controller.model.cube = copy.deepcopy(current_state)
                                    controller.view.root.after(0, lambda: controller.view.update_view(controller.model.cube))
                                    controller.view.root.update_idletasks()
                                    time.sleep(0.05 * len(chosen_alg_moves))
                                print(f"  Algorithm improved score to {score_post_algo}!")
                                action_applied_by_algorithm = True
                                continue
                            else:
                                print(f"  Algorithm didn't improve state, trying individual moves...")
                
                evaluated_action_scores.sort(key=lambda x: x[3], reverse=True) 
                
                action_taken_this_cycle = False 
                score_of_chosen_state = current_score_eval 

                is_in_loop = self._detect_loop(move_history) # Only pass move_history
                if is_in_loop:
                    print("  Loop detected! Applying algorithm breaker...")
                    breaker_moves = self._advanced_loop_breaker(current_state)
                    temp_state_breaker = copy.deepcopy(current_state)
                    for b_face, b_dir in breaker_moves:
                        temp_state_breaker = self._apply_action(temp_state_breaker, b_face, b_dir)
                        solution_moves.append((b_face, b_dir))
                        move_history.append((b_face, b_dir))
                    current_state = temp_state_breaker
                    visited_states.add(self._get_state_hash(current_state))
                    action_taken_this_cycle = True
                    if controller and hasattr(controller.view, 'root'):
                        controller.model.cube = copy.deepcopy(current_state)
                        controller.view.root.after(0, lambda: controller.view.update_view(controller.model.cube))
                        controller.view.root.update_idletasks()
                        time.sleep(0.05 * len(breaker_moves))
                else:
                    chosen_action_details = None
                    if random.random() < exploration_probability and len(evaluated_action_scores) > 1:
                        explorable_candidates = evaluated_action_scores[1:min(6, len(evaluated_action_scores))]
                        if explorable_candidates:
                            adj_scores = np.array([data[3] for data in explorable_candidates], dtype=float)
                            stable_adj_scores = adj_scores - np.max(adj_scores)
                            exp_adj_scores = np.exp(stable_adj_scores)
                            probabilities = exp_adj_scores / np.sum(exp_adj_scores)
                            if np.isnan(probabilities).any() or np.sum(probabilities) == 0: 
                                probabilities = np.ones(len(explorable_candidates)) / len(explorable_candidates)
                            
                            chosen_idx = np.random.choice(len(explorable_candidates), p=probabilities)
                            chosen_action_details = explorable_candidates[chosen_idx]
                            print(f"  Exploring (softmax): {chosen_action_details[0]} (Score: {chosen_action_details[2]}, Adj: {chosen_action_details[3]})")
                    
                    if not chosen_action_details: 
                        if evaluated_action_scores:
                            chosen_action_details = evaluated_action_scores[0]
                            print(f"  Taking best action: {chosen_action_details[0]} (Score: {chosen_action_details[2]}, Adj: {chosen_action_details[3]})")
                        else:
                            print("ERROR: No actions available for selection.")
                            # Add current state hash to history before returning
                            state_hash_history.append(self._get_state_hash(current_state))
                            if len(state_hash_history) > state_hash_history_limit: state_hash_history.pop(0)
                            return solution_moves 

                    selected_action, new_state_selected, original_score_selected, _ = chosen_action_details
                    current_state = new_state_selected
                    visited_states.add(self._get_state_hash(current_state))
                    solution_moves.append(selected_action)
                    move_history.append(selected_action)
                    action_taken_this_cycle = True
                    score_of_chosen_state = original_score_selected 

                    if controller and hasattr(controller.view, 'root'):
                        controller.model.cube = copy.deepcopy(current_state)
                        controller.view.root.after(0, lambda: controller.view.update_view(controller.model.cube))
                        controller.view.root.update_idletasks()
                        time.sleep(0.01) 

                if action_taken_this_cycle: # This applies to loop breaker or individual move
                    if len(move_history) > history_limit:
                        move_history.pop(0)
                    # visited_states.add is handled where current_state is set

                # Append the hash of the current_state (finalized for this step) to state_hash_history
                # This hash will be used by _detect_loop in the *next* step.
                state_hash_history.append(self._get_state_hash(current_state))
                if len(state_hash_history) > state_hash_history_limit:
                    state_hash_history.pop(0)

                if self._is_solved(current_state):
                    print(f"Solution found in {step + 1} steps!")
                    if controller and hasattr(controller.view, 'root'):
                        controller.model.cube = copy.deepcopy(current_state)
                        controller.view.root.after(0, lambda: controller.view.update_view(controller.model.cube))
                    return solution_moves
                
                # Progress tracking: use action_taken_this_cycle or action_applied_by_algorithm
                # The original logic used action_taken_this_cycle. We should ensure this covers algorithms too if they don't 'continue'.
                # However, algorithms currently 'continue', so progress is based on the state after the algo.
                # The score_of_chosen_state is for individual moves.
                # Let's use current_score_eval (score after any action/revert/algo) vs best_score_overall
                
                current_final_score_for_step = self._evaluate_state(current_state) # Re-evaluate after any action
                if current_final_score_for_step > best_score_overall:
                    best_score_overall = current_final_score_for_step
                    best_state_overall = copy.deepcopy(current_state)
                    steps_without_progress = 0
                    print(f"  Found better state! Score: {best_score_overall}")
                elif action_taken_this_cycle or action_applied_by_algorithm : # only increment if an action was attempted
                    steps_without_progress += 1
                
                if steps_without_progress > 15: 
                    exploration_probability = min(0.3, exploration_probability * 1.2) 
                    steps_without_progress = 0 
                    local_optima_counter += 1
                    print(f"  No progress for {15} steps, increasing exploration to {exploration_probability:.2f}")
                    
                    if random.random() < 0.5: 
                        print("  Trying a random common algorithm to break out of local optima")
                        if common_algorithms:
                            chosen_alg_name, chosen_alg_moves = random.choice(common_algorithms)
                            print(f"  Selected algorithm: {chosen_alg_name}")
                            
                            temp_state_optima = copy.deepcopy(current_state)
                            for move in chosen_alg_moves:
                                temp_state_optima = self._apply_action(temp_state_optima, move[0], move[1])
                                solution_moves.append(move)
                                move_history.append(move) # Add to move_history
                            current_state = temp_state_optima
                            visited_states.add(self._get_state_hash(current_state)) # Add to visited
                            
                            if controller and hasattr(controller.view, 'root'):
                                controller.model.cube = copy.deepcopy(current_state)
                                controller.view.root.after(0, lambda: controller.view.update_view(controller.model.cube))
                                controller.view.root.update_idletasks()
                                time.sleep(0.05 * len(chosen_alg_moves))
                    
                    if controller and hasattr(controller.view, 'root'): 
                        controller.view.root.update() 

                    if white_cross_solved: 
                        current_stage, current_target, _ = self._get_next_target_after_cross(current_state)
            
            print(f"Maximum steps reached without solution.")
            return solution_moves
            
        except Exception as e:
            print(f"Error in solve method: {str(e)}")
            import traceback
            traceback.print_exc()
            return []