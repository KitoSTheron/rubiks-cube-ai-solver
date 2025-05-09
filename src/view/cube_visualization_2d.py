import tkinter as tk

class CubeVisualization2D(tk.Frame):
    """
    2D visual representation of the Rubik's Cube state with colored squares
    """
    
    def __init__(self, parent, **kwargs):
        """Initialize the 2D cube visualization"""
        super().__init__(parent, **kwargs)
        
        # Color mapping
        self.color_map = {
            'y': '#FFFF00',  # Yellow
            'b': '#0000FF',  # Blue
            'r': '#FF0000',  # Red
            'o': '#FFA500',  # Orange
            'g': '#00FF00',  # Green
            'w': '#FFFFFF',  # White
        }
        
        # Create canvas for drawing the cube
        self.canvas = tk.Canvas(self, width=400, height=400, bg='#EEEEEE')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initialize the visualization
        self.create_empty_grid()
    
    def create_empty_grid(self):
        """Create an empty visualization grid"""
        # Calculate square size based on canvas size
        self.square_size = 30
        self.spacing = 5
        
        # Define positions for the unfolded cube layout
        # The layout looks like:
        #     [U]
        #  [L][F][R][B]
        #     [D]
        self.face_positions = {
            'U': (1, 0),  # Top face (x=1, y=0)
            'L': (0, 1),  # Left face (x=0, y=1)
            'F': (1, 1),  # Front face (x=1, y=1)
            'R': (2, 1),  # Right face (x=2, y=1)
            'B': (3, 1),  # Back face (x=3, y=1)
            'D': (1, 2),  # Down face (x=1, y=2)
        }
    
    def update(self, cube_data, face_mapping):
        """Update the cube visualization with new data"""
        self.canvas.delete("all")  # Clear previous visualization
        
        # Restore original spacing
        self.spacing = 5  # Back to the original spacing value
        self.square_size = 30  # Keep the same size for squares
        
        # Map the indices to face letters
        index_to_face = {}
        for face_letter, face_idx in face_mapping.items():
            index_to_face[face_idx] = face_letter
        
        # Face name mapping to full words
        face_name_full = {
            'U': 'TOP',
            'L': 'LEFT',
            'R': 'RIGHT',
            'B': 'BACK',
            'D': 'BOTTOM'
            # 'F' removed to skip drawing the FRONT label
        }
        
        # Custom offsets for labels to avoid overlap
        label_offsets = {
            'U': {'x': 0, 'y': 0},
            'L': {'x': 0, 'y': 0},
            'R': {'x': 0, 'y': 0},
            'B': {'x': 0, 'y': 0},
            'D': {'x': 0, 'y': 0}
        }
        
        # Draw each face
        for face_idx, face_letter in index_to_face.items():
            if face_letter not in self.face_positions:
                continue
                
            grid_x, grid_y = self.face_positions[face_letter]
            base_x = grid_x * (3 * self.square_size + self.spacing) + 60  # Keep the increased base offset
            base_y = grid_y * (3 * self.square_size + self.spacing) + 60  # Keep the increased base offset
            
            # Draw the 3x3 grid for this face
            for row in range(3):
                for col in range(3):
                    # Get the color code (e.g., 'y0', 'b1', etc.)
                    sticker = cube_data[face_idx][row][col]
                    color_code = sticker[0]  # First character is the color
                    
                    # Convert to hex color
                    color = self.color_map.get(color_code, '#CCCCCC')
                    
                    # Calculate position
                    x = base_x + col * self.square_size
                    y = base_y + row * self.square_size
                    
                    # Draw the colored square
                    self.canvas.create_rectangle(
                        x, y, 
                        x + self.square_size, y + self.square_size,
                        fill=color,
                        outline='black',
                        width=1
                    )
                    
                    # Draw sticker ID (e.g., '0', '1', etc.)
                    sticker_id = sticker[1:]  # Everything after the color code
                    self.canvas.create_text(
                        x + self.square_size/2,
                        y + self.square_size/2,
                        text=sticker_id,
                        font=("Arial", 10),
                        fill='black' if color_code in ['y', 'w', 'g'] else 'white'
                    )
            
            # Skip drawing the FRONT label
            if face_letter == 'F':
                continue
                
            # Draw face name with custom offset to avoid overlap
            full_name = face_name_full[face_letter]
            offset_x = label_offsets[face_letter]['x']
            offset_y = label_offsets[face_letter]['y']
            
            # Position text based on face location to avoid overlaps
            if face_letter == 'U':
                # TOP label - above the face
                text_y = base_y - 15 + offset_y
                text_x = base_x + 1.5 * self.square_size + offset_x
            elif face_letter == 'D':
                # BOTTOM label - below the face
                text_y = base_y + 3 * self.square_size + 15 + offset_y
                text_x = base_x + 1.5 * self.square_size + offset_x
            elif face_letter in ['L', 'R', 'B']:
                # Side faces - position to the sides or corners as appropriate
                text_y = base_y - 15 + offset_y
                text_x = base_x + 1.5 * self.square_size + offset_x
            
            self.canvas.create_text(
                text_x, text_y,
                text=full_name,
                font=("Arial", 12, "bold")
            )