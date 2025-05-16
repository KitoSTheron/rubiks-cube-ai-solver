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

The `RLCubeSolver` class is designed to find a sequence of moves to solve a scrambled Rubik's Cube. It employs principles of Reinforcement Learning, specifically aiming for a Deep Q-Learning (DQN) approach (though the TensorFlow model integration is a placeholder in the provided code).

**Key Aspects of the RL Algorithm:**

1.  **State Representation (`_state_to_features`):**
    *   The solver needs to convert the cube's matrix representation into a numerical format (features) that a neural network can process. The current code outlines a placeholder for one-hot encoding each sticker's color.

2.  **Actions:**
    *   The agent can perform 18 fundamental actions: rotating any of the 6 faces (Up, Down, Left, Right, Front, Back) in 3 directions (clockwise, counter-clockwise, or a double 180-degree turn).

3.  **Reward System (`_evaluate_state`, `_evaluate_white_cross`, `_evaluate_white_corners`):**
    *   The agent learns by receiving rewards based on the state of the cube. The `_evaluate_state` method calculates a score reflecting how close the cube is to being solved.
    *   It uses a **progressive reward strategy**, meaning it rewards the agent for achieving intermediate goals, such as:
        *   Correctly forming the **white cross** on the bottom face.
        *   Correctly placing and orienting the **white corners** after the cross is complete.
        *   Solving the **First Two Layers (F2L)**.
        *   **Orienting the Last Layer (OLL)**.
        *   **Permuting the Last Layer (PLL)**.
    *   Achieving these milestones yields significant rewards, guiding the agent through the solving stages.

4.  **Solving Process (`solve` method):**
    *   The `solve` method is the main loop where the agent interacts with the cube environment.
    *   **Iterative Improvement:** It iteratively selects and applies moves or algorithms to the current cube state.
    *   **Action Selection:**
        *   The agent evaluates potential moves by predicting their Q-values (using the placeholder model) or by exploring new actions.
        *   It can also choose to apply **common Rubik's Cube algorithms** (`_get_common_algorithms`) as macro-actions. These are predefined sequences of moves known to solve specific sub-problems (e.g., OLL, PLL cases).
    *   **Progress Tracking and Reversion:**
        *   The solver tracks the best score achieved for various stages (e.g., white cross, white corners).
        *   If a series of moves significantly degrades the cube's state (e.g., breaks a completed white cross), the solver can revert to a previously saved, better state to avoid getting stuck in unproductive paths.
    *   **Loop Detection and Breaking (`_detect_loop`, `_advanced_loop_breaker`):**
        *   To prevent the agent from repeating the same sequence of moves endlessly, the `_detect_loop` function checks the history of moves and states.
        *   If a loop is detected, the `_advanced_loop_breaker` function applies a "breaker" sequence of moves (often a short, somewhat random algorithm) to try and perturb the state and escape the local optimum.
    *   **Termination:** The process continues until the cube is solved, a maximum number of steps is reached, or a maximum runtime is exceeded.

5.  **Exploration and Exploitation:**
    *   The agent balances exploring new, unknown sequences of moves with exploiting known good moves (based on its learned Q-values or heuristic evaluations). An `exploration_probability` parameter controls this balance.

The overall strategy is to guide the RL agent through the complex state space of the Rubik's Cube by breaking the problem down into manageable sub-goals, rewarding progress at each stage, and using techniques to avoid getting stuck.

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