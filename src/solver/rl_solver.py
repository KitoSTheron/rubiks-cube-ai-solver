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
            ("R Move", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise")]),
            ("Sledgehammer", [("R", "clockwise"), ("F", "clockwise"), ("R", "counterclockwise"), ("F", "counterclockwise")]),
            ("L Move", [("R", "clockwise"), ("U", "clockwise"), ("R", "counterclockwise"), ("U", "counterclockwise")] * 2),
            # Add more common algorithms here
        ]
        return algorithms

    def _evaluate_state(self, cube_state):
        """Evaluate how close the cube is to being solved with a more flexible approach"""
        score = 0
        
        # --- Basic scoring for all stickers ---
        # For each face, give points for matching the center
        for face_idx in range(len(cube_state)):
            # Get the center color
            center_color = cube_state[face_idx][1][1][0]
            
            # Count stickers matching the center
            for row in range(3):
                for col in range(3):
                    if cube_state[face_idx][row][col][0] == center_color:
                        score += 1
        
        # --- Bonus for complete faces ---
        for face_idx in range(len(cube_state)):
            center_color = cube_state[face_idx][1][1][0]
            face_complete = True
            
            for row in range(3):
                for col in range(3):
                    if cube_state[face_idx][row][col][0] != center_color:
                        face_complete = False
                        break
                if not face_complete:
                    break
            
            if face_complete:
                score += 15  # Significant bonus for a complete face
        
        # --- WHITE CROSS DETECTION (more flexible) ---
        bottom_face = 5  # White face
        front_face = 1   # Blue face
        right_face = 2   # Red face
        left_face = 3    # Orange face
        back_face = 4    # Green face
        
        # Check white center
        if cube_state[bottom_face][1][1][0] == 'w':
            score += 5
            
            # Check white edges, giving partial credit
            white_edge_positions = [(0, 1), (1, 0), (1, 2), (2, 1)]  # top, left, right, bottom edges
            adjacent_faces = [front_face, left_face, right_face, back_face]
            adjacent_pos = [(2, 1), (2, 1), (2, 1), (2, 1)]  # positions on adjacent faces
            
            white_edges_correct = 0
            
            for i, pos in enumerate(white_edge_positions):
                row, col = pos
                adj_row, adj_col = adjacent_pos[i]
                adj_face = adjacent_faces[i]
                
                # Give points for having white on the bottom face
                if cube_state[bottom_face][row][col][0] == 'w':
                    score += 2
                    
                    # Give additional points for correct adjacent color
                    if cube_state[adj_face][adj_row][adj_col][0] == ['b', 'o', 'r', 'g'][i]:
                        score += 3
                        white_edges_correct += 1
            
            # Bonus for complete white cross
            if white_edges_correct == 4:
                score += 10
                
                # --- F2L DETECTION (First Two Layers) ---
                # Check white corners
                corner_positions = [(0, 0), (0, 2), (2, 0), (2, 2)]  # corners
                white_corners_correct = 0
                
                for corner_pos in corner_positions:
                    row, col = corner_pos
                    if cube_state[bottom_face][row][col][0] == 'w':
                        score += 2
                        # We'd need to check adjacent faces for complete correctness
                        # This is simplified for now
                
                # Check middle layer edges (simplified)
                middle_edges_correct = 0
                # Front-left, front-right, back-left, back-right
                middle_positions = [(front_face, 1, 0), (front_face, 1, 2), 
                                    (back_face, 1, 2), (back_face, 1, 0)]
                adjacent_middle = [(left_face, 1, 2), (right_face, 1, 0),
                                 (left_face, 1, 0), (right_face, 1, 2)]
                
                for i in range(4):
                    face, row, col = middle_positions[i]
                    adj_face, adj_row, adj_col = adjacent_middle[i]
                    
                    # Give points for matching the center color
                    if cube_state[face][row][col][0] == cube_state[face][1][1][0]:
                        score += 2
                        
                        # Additional points if adjacent color matches too
                        if cube_state[adj_face][adj_row][adj_col][0] == cube_state[adj_face][1][1][0]:
                            score += 3
                            middle_edges_correct += 1
                
                # Bonus for complete F2L
                if middle_edges_correct == 4:
                    score += 15
                    
                    # --- OLL DETECTION (Orient Last Layer) ---
                    top_face = 0  # Yellow face
                    yellow_count = 0
                    
                    for row in range(3):
                        for col in range(3):
                            if cube_state[top_face][row][col][0] == 'y':
                                yellow_count += 1
                                score += 1
                    
                    # Bonus for complete yellow face
                    if yellow_count == 9:
                        score += 20
                        
                        # --- PLL DETECTION (Permute Last Layer) ---
                        # Check if the top layer edges are aligned with their centers
                        top_edge_positions = [(0, 1), (1, 0), (1, 2), (2, 1)]  # top, left, right, bottom
                        adjacent_top_faces = [back_face, left_face, right_face, front_face]
                        adjacent_top_pos = [(0, 1), (0, 1), (0, 1), (0, 1)]
                        
                        top_edges_correct = 0
                        
                        for i, pos in enumerate(top_edge_positions):
                            adj_face = adjacent_top_faces[i]
                            adj_row, adj_col = adjacent_top_pos[i]
                            
                            if cube_state[adj_face][adj_row][adj_col][0] == cube_state[adj_face][1][1][0]:
                                score += 2
                                top_edges_correct += 1
                        
                        # Bonus for complete PLL edges
                        if top_edges_correct == 4:
                            score += 15
        
        return score
    
    def solve(self, cube_state):
        """Solve the cube with a better balance of exploration and exploitation"""
        print("Attempting to solve cube...")
        
        # If the cube is already solved, return empty solution
        if self._is_solved(cube_state):
            print("Cube is already solved!")
            return []
        
        # Deep copy to avoid modifying the original
        current_state = copy.deepcopy(cube_state)
        solution_moves = []
        
        # Track the best state we've seen
        best_score_overall = self._evaluate_state(current_state)
        best_state_overall = copy.deepcopy(current_state)
        
        # Parameters with better tuning
        exploration_probability = 0.1  # Lower initial exploration probability
        local_optima_counter = 0       
        local_optima_threshold = 5     # Detect local optima sooner
        max_steps_without_progress = 20 # Allow fewer non-improving steps
        steps_without_progress = 0
        
        # Get common algorithms
        common_algorithms = self._get_common_algorithms()
        
        for step in range(self.max_steps):
            print(f"Step {step + 1}/{self.max_steps}")
            current_score = self._evaluate_state(current_state)
            
            # Track if we found a better state in this iteration
            found_better_state = False
            
            # If we're potentially stuck in local optima, slightly increase exploration
            if local_optima_counter > local_optima_threshold:
                exploration_probability = min(0.3, exploration_probability * 1.2)  # More conservative increase
                print(f"  Increasing exploration to {exploration_probability:.2f}")
                local_optima_counter = 0
            
            # Try each possible action
            action_scores = []
            for action in self.actions:
                face, direction = action
                
                # Apply the action and evaluate the resulting state
                new_state = self._apply_action(current_state, face, direction)
                score = self._evaluate_state(new_state)
                action_scores.append((action, new_state, score))
            
            # Sort actions by score (best first)
            action_scores.sort(key=lambda x: x[2], reverse=True)
            
            # With 70% probability, just take the best action (more exploitation)
            if random.random() > exploration_probability:
                # Choose the best action
                best_action, best_state, best_score = action_scores[0]
                print(f"  Taking best action: {best_action[0]} {best_action[1]} (score: {best_score})")
            else:
                # Weighted random selection based on scores (smarter exploration)
                max_score = max(score for _, _, score in action_scores)
                min_score = min(score for _, _, score in action_scores)
                score_range = max(1, max_score - min_score)
                
                # More aggressive weighting - favor higher scores more strongly
                weights = [((score - min_score) / score_range) ** 3 + 0.01 for _, _, score in action_scores]
                total_weight = sum(weights)
                probs = [w / total_weight for w in weights]
                
                chosen_idx = np.random.choice(len(action_scores), p=probs)
                best_action, best_state, best_score = action_scores[chosen_idx]
                print(f"  Exploring: {best_action[0]} {best_action[1]} (score: {best_score}, rank: {chosen_idx+1}/{len(action_scores)})")
            
            # Apply the selected action
            face, direction = best_action
            current_state = best_state
            solution_moves.append(best_action)
            
            # Check if we've reached a solution
            if self._is_solved(current_state):
                print(f"Solution found in {step + 1} steps!")
                return solution_moves
            
            # Check if we've found a better overall state
            if best_score > best_score_overall:
                best_score_overall = best_score
                best_state_overall = copy.deepcopy(current_state)
                steps_without_progress = 0
                found_better_state = True
                print(f"  Found better state! Score: {best_score}")
            else:
                steps_without_progress += 1
            
            # Always save any state that matches the best score we've seen
            if best_score == best_score_overall:
                best_state_overall = copy.deepcopy(current_state)
                steps_without_progress = max(0, steps_without_progress - 1)  # Reduce non-progress counter
            
            # If we haven't made progress for too long, revert to the best known state
            if steps_without_progress > max_steps_without_progress:
                print(f"  No progress for {steps_without_progress} steps, reverting to best known state")
                current_state = copy.deepcopy(best_state_overall)
                steps_without_progress = 0
                # Reset exploration to be lower after reverting
                exploration_probability = max(0.05, exploration_probability * 0.7)
            
            # Try common algorithms less frequently and only when needed
            if (local_optima_counter > local_optima_threshold or 
                (step % 15 == 0 and steps_without_progress > 5)):
                print("  Trying a common algorithm...")
                best_algorithm_score = -float('inf')
                best_algorithm_name = None
                best_algorithm = None
                best_algorithm_state = None
                
                for name, algorithm in common_algorithms:
                    new_state = self._apply_algorithm(current_state, algorithm)
                    score = self._evaluate_state(new_state)
                    
                    if score > best_algorithm_score:
                        best_algorithm_score = score
                        best_algorithm_name = name
                        best_algorithm = algorithm
                        best_algorithm_state = new_state
                
                # Only apply the algorithm if it actually improves the state or gives the best score so far
                if best_algorithm_score > current_score or best_algorithm_score >= best_score_overall:
                    print(f"  Applying algorithm {best_algorithm_name} (score: {best_algorithm_score})")
                    current_state = best_algorithm_state
                    solution_moves.extend(best_algorithm)
                    
                    # Update best state if this algorithm gave us a new best score
                    if best_algorithm_score > best_score_overall:
                        best_score_overall = best_algorithm_score
                        best_state_overall = copy.deepcopy(current_state)
                        steps_without_progress = 0
                        found_better_state = True
                    
                    continue
            
            # Update local optima counter
            if not found_better_state:
                local_optima_counter += 1
            else:
                local_optima_counter = 0
        
        print(f"Maximum steps reached without solution. Found {len(solution_moves)} moves.")
        # Return the solution that led to the best state we found
        return solution_moves

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

