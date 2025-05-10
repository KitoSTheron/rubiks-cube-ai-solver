import tkinter as tk
from tkinter import ttk
from view.cube_visualization_2d import CubeVisualization2D

class MainWindow:
    """
    Main application window.
    Manages the UI components and layout.
    """
    
    def __init__(self, controller):
        """
        Initialize the main window
        
        Args:
            controller: The application controller
        """
        self.controller = controller
        self.root = tk.Tk()
        self.root.title("Rubik's Cube Solver")
        self.root.geometry("1000x800")  # Increased size to fit the cube state display
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the UI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title label
        title_label = ttk.Label(main_frame, text="Rubik's Cube Controls", font=("Arial", 16))
        title_label.pack(pady=10)
        
        # Layout with left side for controls and right side for cube state
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Controls
        controls_frame = ttk.LabelFrame(content_frame, text="Controls")
        controls_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Right side - Split into two sections
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Upper section - 2D Visualization
        viz_frame = ttk.LabelFrame(right_frame, text="2D Cube Visualization")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create the 2D visualization
        self.cube_viz_2d = CubeVisualization2D(viz_frame)
        self.cube_viz_2d.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Lower section - Text representation
        cube_state_frame = ttk.LabelFrame(right_frame, text="Cube State Text")
        cube_state_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a frame for the cube visualization
        self.cube_text = tk.Text(cube_state_frame, width=50, height=20, font=("Courier", 10))
        self.cube_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create rotation button frames for each face
        faces = [
            {"name": "U (Top)", "rotations": ["U", "U'", "U2"]},
            {"name": "D (Bottom)", "rotations": ["D", "D'", "D2"]},
            {"name": "L (Left)", "rotations": ["L", "L'", "L2"]},
            {"name": "R (Right)", "rotations": ["R", "R'", "R2"]},
            {"name": "F (Front)", "rotations": ["F", "F'", "F2"]},
            {"name": "B (Back)", "rotations": ["B", "B'", "B2"]}
        ]
        
        # Create a dictionary to map button text to rotation commands
        self.rotation_commands = {
            "U": lambda: self.controller.rotate_face("U", "clockwise"),
            "U'": lambda: self.controller.rotate_face("U", "counterclockwise"),
            "U2": lambda: self.controller.rotate_face("U", "double"),
            "D": lambda: self.controller.rotate_face("D", "clockwise"),
            "D'": lambda: self.controller.rotate_face("D", "counterclockwise"),
            "D2": lambda: self.controller.rotate_face("D", "double"),
            "L": lambda: self.controller.rotate_face("L", "clockwise"),
            "L'": lambda: self.controller.rotate_face("L", "counterclockwise"),
            "L2": lambda: self.controller.rotate_face("L", "double"),
            "R": lambda: self.controller.rotate_face("R", "clockwise"),
            "R'": lambda: self.controller.rotate_face("R", "counterclockwise"),
            "R2": lambda: self.controller.rotate_face("R", "double"),
            "F": lambda: self.controller.rotate_face("F", "clockwise"),
            "F'": lambda: self.controller.rotate_face("F", "counterclockwise"),
            "F2": lambda: self.controller.rotate_face("F", "double"),
            "B": lambda: self.controller.rotate_face("B", "clockwise"),
            "B'": lambda: self.controller.rotate_face("B", "counterclockwise"),
            "B2": lambda: self.controller.rotate_face("B", "double")
        }
        
        # Create rotation buttons
        rotation_controls = ttk.Frame(controls_frame)
        rotation_controls.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for i, face in enumerate(faces):
            face_frame = ttk.LabelFrame(rotation_controls, text=face["name"])
            face_frame.grid(row=i//3, column=i%3, padx=10, pady=5, sticky="ew")
            
            for j, rotation in enumerate(face["rotations"]):
                btn = ttk.Button(
                    face_frame, 
                    text=rotation,
                    width=3,
                    command=self.rotation_commands[rotation]
                )
                btn.grid(row=0, column=j, padx=5, pady=5)
        
        # Configure grid weights for rotation controls
        for i in range(2):
            rotation_controls.grid_rowconfigure(i, weight=1)
        for i in range(3):
            rotation_controls.grid_columnconfigure(i, weight=1)
        
        # Add utility buttons at the bottom
        utility_frame = ttk.Frame(controls_frame)
        utility_frame.pack(fill=tk.X, pady=10, padx=10)
        
        reset_btn = ttk.Button(utility_frame, text="Reset Cube", command=self.controller.reset_cube)
        reset_btn.pack(side=tk.LEFT, padx=10)
        
        scramble_btn = ttk.Button(utility_frame, text="Scramble", command=self.controller.scramble_cube)
        scramble_btn.pack(side=tk.LEFT, padx=10)
        
        # Add runtime configuration and custom solve button
        runtime_frame = ttk.Frame(controls_frame)
        runtime_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(runtime_frame, text="Max Runtime (sec):").pack(side=tk.LEFT)
        self.runtime_var = tk.StringVar(value="300")  # Default 5 minutes
        runtime_entry = ttk.Entry(runtime_frame, textvariable=self.runtime_var, width=5)
        runtime_entry.pack(side=tk.LEFT, padx=5)

        # Create a single solve button that uses the runtime from the entry field
        solve_button = ttk.Button(
            controls_frame, 
            text="Solve Cube", 
            command=lambda: self.controller.solve_cube(max_runtime=int(self.runtime_var.get()))
        )
        solve_button.pack(fill=tk.X, padx=10, pady=5)
    
    def run(self):
        """Run the main event loop"""
        # Update the cube state before starting the event loop
        self.update_view(self.controller.model.cube)
        self.root.mainloop()
    
    def update_view(self, data):
        """
        Update the view with new data
        
        Args:
            data: The cube state to display
        """
        # Clear the current text
        self.cube_text.delete('1.0', tk.END)
        
        # Format and display the cube state like in debug_cube_state
        model = self.controller.model
        face_names = {
            model.top_face: "TOP (Yellow)",
            model.front_face: "FRONT (Blue)",
            model.right_face: "RIGHT (Red)",
            model.left_face: "LEFT (Orange)",
            model.back_face: "BACK (Green)",
            model.bottom_face: "BOTTOM (White)"
        }
        
        # Create a mapping from face letters to face indices for the visualization
        face_mapping = {
            'U': model.top_face,
            'L': model.left_face,
            'F': model.front_face,
            'R': model.right_face,
            'B': model.back_face,
            'D': model.bottom_face
        }
        
        # Title
        self.cube_text.insert(tk.END, "--- Current Cube State ---\n\n", "title")
        
        for idx, face_name in face_names.items():
            self.cube_text.insert(tk.END, f"{face_name}:\n", "face_title")
            for row in data[idx]:
                self.cube_text.insert(tk.END, f"  {row}\n", "face_data")
            self.cube_text.insert(tk.END, "\n")
        
        self.cube_text.insert(tk.END, "-" * 40 + "\n", "separator")
        
        # Set text as read-only
        self.cube_text.config(state=tk.DISABLED)
        
        # Configure tags for styling
        self.cube_text.tag_configure("title", font=("Arial", 12, "bold"))
        self.cube_text.tag_configure("face_title", font=("Arial", 10, "bold"))
        self.cube_text.tag_configure("face_data", font=("Courier", 10))
        self.cube_text.tag_configure("separator", font=("Arial", 10))
        
        # Re-enable for the next update
        self.cube_text.config(state=tk.NORMAL)
        
        # Update the 2D visualization
        self.cube_viz_2d.update(data, face_mapping)