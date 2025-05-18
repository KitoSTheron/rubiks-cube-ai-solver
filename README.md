# Rubik's Cube AI Solver

This project implements a Rubik's Cube solver using Reinforcement Learning (RL) with a graphical user interface built using Tkinter. The application follows the Model-View-Controller (MVC) design pattern to organize its components.

## How it Works

The system combines a visual representation of the Rubik's Cube with an AI agent that learns to solve it. The core logic is divided into how the application is structured (MVC), how the cube itself is represented and manipulated, and how the AI learns to solve it.

### MVC Design Pattern

The application is structured using the Model-View-Controller (MVC) architectural pattern:

*   **Model (`src/model/data_model.py`):**
    *   This component is responsible for managing the data and business logic of the application.
    *   The `DataModel` class holds the current state of the Rubik's Cube, its solved state, and methods to perform rotations on the cube's faces. It directly manipulates the cube's data structure.

*   **View (`src/view/main_window.py`, `src/view/cube_visualization_2d.py`):**
    *   This component handles the presentation of data to the user.
    *   `MainWindow` sets up the main application window, including buttons for user interaction (rotate, scramble, solve, reset).
    *   `CubeVisualization2D` provides a 2D unfolded representation of the Rubik's Cube, rendering the colors of each sticker. It updates based on the data from the Model.

*   **Controller (`src/controller/app_controller.py`):**
    *   This component acts as an intermediary between the Model and the View.
    *   The `AppController` class initializes the Model and View. It handles user input from the View (e.g., button clicks), processes these inputs, instructs the Model to update its state (e.g., perform a rotation), and then tells the View to refresh its display to reflect the changes in the Model. It also initiates the cube solving process by invoking the `RLCubeSolver`.

### Cube Representation (`DataModel`)

The Rubik's Cube state is represented in the `DataModel` class as a list of six 3x3 matrices. Each matrix corresponds to one face of the cube:

```python
# self.cube in DataModel
self.cube = [
    [['y0', 'y1', 'y2'], ['y3', 'y4', 'y5'], ['y6', 'y7', 'y8']], # 0: Top face (Yellow)
    [['b0', 'b1', 'b2'], ['b3', 'b4', 'b5'], ['b6', 'b7', 'b8']], # 1: Front face (Blue)
    [['r0', 'r1', 'r2'], ['r3', 'r4', 'r5'], ['r6', 'r7', 'r8']], # 2: Right face (Red)
    [['o0', 'o1', 'o2'], ['o3', 'o4', 'o5'], ['o6', 'o7', 'o8']], # 3: Left face (Orange)
    [['g0', 'g1', 'g2'], ['g3', 'g4', 'g5'], ['g6', 'g7', 'g8']], # 4: Back face (Green)
    [['w0', 'w1', 'w2'], ['w3', 'w4', 'w5'], ['w6', 'w7', 'w8']]  # 5: Bottom face (White)
]
```

*   Each element within these matrices is a string (e.g., `'y0'`, `'b1'`). The first character denotes the color of the sticker (e.g., 'y' for yellow), and the subsequent character(s) act as a unique identifier for that sticker position.
*   The `DataModel` contains methods for each fundamental face rotation (e.g., `rotate_R()` for a clockwise turn of the right face, `rotate_U_prime()` for a counter-clockwise turn of the upper face). These methods directly manipulate the list of matrices by:
    1.  Rotating the stickers on the face itself.
    2.  Updating the affected stickers on the four adjacent faces.

### Reinforcement Learning Solver (`src/solver/rl_solver.py`)

The `RLCubeSolver` class is designed to find a sequence of moves to solve a scrambled Rubik's Cube. It employs principles of Reinforcement Learning (RL) with a focus on Deep Q-Learning (DQN). The solver integrates heuristic evaluations, dynamic action filtering, and enhanced state retention to ensure efficient and structured solving.

**Key Features of the RL Solver:**

1. **State Representation (`_state_to_features`):**
   - The solver converts the cube's matrix representation into a numerical format (features) that a neural network can process. This is achieved using one-hot encoding for each sticker's color.

2. **Dynamic Action Filtering:**
   - The solver dynamically adjusts the available actions (`self.actions`) based on the current solving stage:
     - **Cross**: Only single moves are allowed.
     - **F2L**: Single moves and F2L algorithms are allowed.
     - **OLL**: Single moves, OLL algorithms, and U moves are allowed.
     - **PLL**: Single moves, PLL algorithms, and U moves are allowed.
   - This ensures the solver focuses on relevant moves and algorithms for each stage, improving efficiency and reducing unnecessary exploration.

3. **Reward System:**
   - The solver uses a **progressive reward strategy**, rewarding the agent for achieving intermediate goals:
     - **White Cross**: Rewards for correctly placing and orienting the white edges.
     - **White Corners**: Rewards for correctly placing and orienting the white corners.
     - **F2L**: Rewards for correctly placing the middle-layer edges.
     - **OLL**: Rewards for orienting all stickers on the top face to yellow.
     - **PLL**: Rewards for permuting all pieces on the top face to their correct positions.
   - Each stage requires the previous stage to be fully complete before scoring is allowed, ensuring a structured and logical solving process.

4. **Progress Tracking and Retention of Good States:**
   - The solver tracks the best state and score for each stage (e.g., white cross, white corners, F2L, OLL, PLL).
   - If significant progress is lost (e.g., regression in the white cross or corners), the solver reverts to the best-known state for that stage.
   - Regression counters allow temporary setbacks but prevent prolonged loss of progress.

5. **Learning and Experience Replay:**
   - The solver uses a replay buffer to store experiences (`state`, `action`, `reward`, `next_state`, `done`) during the solving process.
   - A minibatch of experiences is sampled from the buffer to train the neural network, ensuring the agent learns from past actions.
   - The Q-values predicted by the neural network are blended with heuristic evaluations to guide action selection.

6. **Action Selection:**
   - The solver uses an epsilon-greedy strategy for action selection:
     - With probability `epsilon`, the solver explores by selecting a random action.
     - Otherwise, it exploits by selecting the action with the highest Q-value or heuristic score.
   - Epsilon is annealed over time, gradually reducing exploration as the solver progresses.


7. **Stage-Specific Focus:**
   - The solver dynamically updates its focus and action set based on the current solving stage:
     - **Cross**: Prioritizes speed and single moves.
     - **F2L**: Focuses on placing middle-layer edges using F2L algorithms.
     - **OLL**: Focuses on orienting the last layer using OLL algorithms and U moves.
     - **PLL**: Focuses on permuting the last layer using PLL algorithms and U moves.

8. **Termination:**
   - The solving process continues until the cube is solved, a maximum number of steps is reached, or a maximum runtime is exceeded.

---

### Example of Progressive Scoring:
- **White Cross**:
  - Each correctly placed and oriented edge: +25 points.
  - Bonus for completing the cross: +50 points.
- **White Corners**:
  - Each correctly placed and oriented corner: +20 points.
  - Bonus for completing all corners: +50 points.
- **F2L**:
  - Each correctly placed middle-layer edge: +25 points.
  - Bonus for completing F2L: +50 points.
- **OLL**:
  - Bonus for orienting all stickers on the top face to yellow: +100 points.
- **PLL**:
  - Bonus for permuting all pieces on the top face to their correct positions: +100 points.
  - Bonus for solving the cube: +200 points.

---

## Running the Application

To run the Rubik's Cube AI Solver, navigate to the project's root directory and execute:

```bash
python src/main.py
```

You can also use the following command-line arguments:

*   `--debug`: Activates a debug flag in the application controller.
*   `--runtime <seconds>`: Specifies the maximum time (in seconds) the solver will run before stopping. The default is 300 seconds (5 minutes).
    Example: `python src/main.py --runtime 600`

## Project Structure

```
rubiks-cube-ai-solver/
├── src/
│   ├── controller/
│   │   └── app_controller.py   # Application controller
│   ├── model/
│   │   └── data_model.py       # Cube data representation and logic
│   ├── solver/
│   │   └── rl_solver.py        # Reinforcement Learning solver
│   ├── view/
│   │   ├── cube_visualization_2d.py # 2D cube drawing
│   │   └── main_window.py      # Main GUI window
│   └── main.py                 # Main application entry point
└── README.md                   # This file
```