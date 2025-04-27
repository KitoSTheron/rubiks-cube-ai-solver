import tkinter as tk
from tkinter import ttk, messagebox
from src.cube.model import Cube
from src.cube.solver import Solver
from src.gui.cube_renderer import CubeRenderer
from src.gui.interaction import Interaction

class MainWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("Rubik's Cube GUI")
        self.master.geometry("1000x600")
        self.cube = Cube()
        self.solver = Solver(self.cube)
        self.setup_ui()

    def setup_ui(self):
        # Create main frame
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for the cube
        self.canvas = tk.Canvas(self.main_frame, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Initialize the cube renderer
        self.cube_renderer = CubeRenderer(self.canvas)
        
        # Initialize interaction handler - now passing the cube object as well
        self.interaction = Interaction(self.cube_renderer, self.cube)
        
        # Bind events to canvas
        self.canvas.bind("<Button-1>", self.interaction.handle_click)
        self.canvas.bind("<B1-Motion>", self.interaction.handle_drag)
        self.canvas.bind("<ButtonRelease-1>", self.interaction.handle_release)
        
        # Create control panel
        self.control_panel = ttk.Frame(self.main_frame, padding=10)
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add controls
        ttk.Label(self.control_panel, text="Rubik's Cube Controls", font=("Arial", 14)).pack(pady=10)
        
        # Reset button
        ttk.Button(self.control_panel, text="Reset Cube", command=self.reset_cube).pack(fill=tk.X, pady=5)
        
        # Solve button
        ttk.Button(self.control_panel, text="Solve Cube", command=self.solve_cube).pack(fill=tk.X, pady=5)
        
        # Scramble button
        ttk.Button(self.control_panel, text="Scramble Cube", command=self.scramble_cube).pack(fill=tk.X, pady=5)
        
        # Face rotation controls
        ttk.Label(self.control_panel, text="Face Rotations", font=("Arial", 12)).pack(pady=(10, 5))
        
        # Create a frame for face rotation buttons
        face_buttons_frame = ttk.Frame(self.control_panel)
        face_buttons_frame.pack(fill=tk.X, pady=5)
        
        # Add face rotation buttons using standard notation
        faces = ['U', 'D', 'L', 'R', 'F', 'B']
        for i, face in enumerate(faces):
            col = i % 2
            row = i // 2
            
            # Frame for each face with CW, CCW, and 180 buttons
            face_frame = ttk.Frame(face_buttons_frame)
            face_frame.grid(row=row, column=col, padx=5, pady=5)
            
            ttk.Label(face_frame, text=f"Face {face}").pack()
            
            btn_frame = ttk.Frame(face_frame)
            btn_frame.pack()
            
            # Clockwise rotation (just the letter)
            ttk.Button(btn_frame, text=f"{face}", width=3, 
                    command=lambda f=face: self.rotate_face(f, 'clockwise')).pack(side=tk.LEFT)
            
            # Counterclockwise rotation (letter prime)
            ttk.Button(btn_frame, text=f"{face}'", width=3,
                    command=lambda f=face: self.rotate_face(f, 'counterclockwise')).pack(side=tk.LEFT)
            
            # 180-degree rotation (letter2)
            ttk.Button(btn_frame, text=f"{face}2", width=3,
                    command=lambda f=face: self.rotate_face(f, '180')).pack(side=tk.LEFT)
        
        # Add swipe handling for mobile-like rotation
        self.canvas.bind("<ButtonRelease-1>", self.interaction.handle_release)
        
        # Draw the initial cube
        self.cube_renderer.draw_cube(self.cube.get_state())

    def reset_cube(self):
        self.cube = Cube()
        self.solver = Solver(self.cube)
        # Update the cube reference in the interaction handler
        self.interaction.cube = self.cube
        self.cube_renderer.draw_cube(self.cube.get_state())

    def solve_cube(self):
        solution = self.solver.solve()
        if solution:
            messagebox.showinfo("Solution", f"Solution found: {' '.join(solution)}")
            # Here you would animate the solution
        else:
            messagebox.showinfo("Already Solved", "The cube is already solved!")

    def scramble_cube(self):
        # Simple scramble functionality
        import random
        faces = ['U', 'D', 'L', 'R', 'F', 'B']
        for _ in range(20):
            face = random.choice(faces)
            self.cube.rotate_face(face)
        self.cube_renderer.draw_cube(self.cube.get_state())

    def rotate_face(self, face, direction):
        """
        Rotate a face of the cube based on standard notation
        face: one of 'U', 'D', 'L', 'R', 'F', 'B'
        direction: 'clockwise', 'counterclockwise', or '180'
        """
        # Call the cube model with the appropriate direction
        self.cube.rotate_face(face, direction)
        
        # Update the display
        self.interaction.update_cube_state()

    def start_event_loop(self):
        self.master.mainloop()