import math

class CubeRenderer:
    def __init__(self, canvas):
        self.canvas = canvas
        self.canvas_width = 800
        self.canvas_height = 600
        self.center_x = self.canvas_width // 2
        self.center_y = self.canvas_height // 2
        
        # 3D rotation angles
        self.rotation_x = 30
        self.rotation_y = 45
        self.rotation_z = 0
        
        # Color mapping - swapped red and orange
        self.colors = {
            'W': "#FFFFFF",  # White
            'O': "#FF0000",  # Orange (was Red)
            'B': "#0000FF",  # Blue
            'R': "#FFA500",  # Red (was Orange)
            'G': "#00FF00",  # Green
            'Y': "#FFFF00"   # Yellow
        }
        
        # Scale factor for the cube
        self.scale = 100
        
        # Cube internal block color
        self.block_color = "#333333"
        
        # Store the canvas items for each face
        self.face_items = {}

    def project_point(self, point3d):
        # Improved 3D to 2D projection with standardized rotation order
        x, y, z = point3d
        
        # Apply rotations in a fixed order (Y-X-Z) which is more intuitive for orbiting
        # First Y-axis rotation (horizontal mouse movement)
        rad_y = math.radians(self.rotation_y)
        x, z = x * math.cos(rad_y) + z * math.sin(rad_y), -x * math.sin(rad_y) + z * math.cos(rad_y)
        
        # Then X-axis rotation (vertical mouse movement)
        rad_x = math.radians(self.rotation_x)
        y, z = y * math.cos(rad_x) - z * math.sin(rad_x), y * math.sin(rad_x) + z * math.cos(rad_x)
        
        # Finally Z-axis rotation (optional, for stability)
        rad_z = math.radians(self.rotation_z)
        x, y = x * math.cos(rad_z) - y * math.sin(rad_z), x * math.sin(rad_z) + y * math.cos(rad_z)
        
        # Scale and translate to canvas coordinates
        x = self.center_x + x * self.scale
        y = self.center_y + y * self.scale
        
        return x, y, z  # Return the projected z for depth sorting

    def draw_cube(self, cube_state):
        # Clear canvas
        self.canvas.delete("all")
        self.face_items = {}
        
        # Calculate all visible faces and their depths - including internal cubie faces
        faces = []
        
        # Define the 6 face directions (normal vectors)
        face_directions = [
            (0, -1, 0),  # Top (U) - White
            (-1, 0, 0),  # Left (L) - Orange
            (0, 0, 1),   # Front (F) - Blue
            (1, 0, 0),   # Right (R) - Red
            (0, 0, -1),  # Back (B) - Green
            (0, 1, 0)    # Bottom (D) - Yellow
        ]
        
        face_indices = {
            (0, -1, 0): 0,  # Top - White
            (-1, 0, 0): 3,  # Left - Orange
            (0, 0, 1): 2,   # Front - Blue
            (1, 0, 0): 1,   # Right - Red
            (0, 0, -1): 4,  # Back - Green
            (0, 1, 0): 5    # Bottom - Yellow
        }
        
        # Draw the internal black cube first as a base
        # For each cubie position
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    if x == 0 and y == 0 and z == 0:  # Skip center cubie
                        continue
                        
                    # For each face of this cubie
                    for nx, ny, nz in face_directions:
                        # Calculate the center of this cubie face
                        cx = x + nx * 0.5
                        cy = y + ny * 0.5
                        cz = z + nz * 0.5
                        
                        # Is this an exterior face?
                        is_exterior = (x == nx and nx != 0) or (y == ny and ny != 0) or (z == nz and nz != 0)
                        
                        # Size of the cubie face - full size
                        size = 0.5
                        
                        # Generate the four corners of the face
                        if nx != 0:  # X-facing face
                            corners = [
                                (cx, cy - size, cz - size),
                                (cx, cy - size, cz + size),
                                (cx, cy + size, cz + size),
                                (cx, cy + size, cz - size)
                            ]
                        elif ny != 0:  # Y-facing face
                            corners = [
                                (cx - size, cy, cz - size),
                                (cx + size, cy, cz - size),
                                (cx + size, cy, cz + size),
                                (cx - size, cy, cz + size)
                            ]
                        else:  # Z-facing face
                            corners = [
                                (cx - size, cy - size, cz),
                                (cx + size, cy - size, cz),
                                (cx + size, cy + size, cz),
                                (cx - size, cy + size, cz)
                            ]
                        
                        # Project corners to 2D and calculate average depth
                        proj_corners = []
                        avg_depth = 0
                        for corner in corners:
                            px, py, pz = self.project_point(corner)
                            proj_corners.append((px, py))
                            avg_depth += pz
                        avg_depth /= 4
                        
                        # Determine the color based on whether it's exterior or interior
                        if is_exterior:
                            # This is an exterior face, use the color from cube_state
                            face_idx = face_indices[(nx, ny, nz)]
                            
                            # Calculate the sticker index within the face
                            if face_idx == 0:  # Top face
                                row = z + 1
                                col = x + 1
                                sticker_idx = row * 3 + col
                            elif face_idx == 5:  # Bottom face
                                row = 1 - z
                                col = x + 1
                                sticker_idx = row * 3 + col
                            elif face_idx == 1:  # Right face
                                row = y + 1
                                col = 1 - z
                                sticker_idx = row * 3 + col
                            elif face_idx == 3:  # Left face
                                row = y + 1
                                col = z + 1
                                sticker_idx = row * 3 + col
                            elif face_idx == 2:  # Front face
                                row = y + 1
                                col = x + 1
                                sticker_idx = row * 3 + col
                            elif face_idx == 4:  # Back face
                                row = y + 1
                                col = 1 - x
                                sticker_idx = row * 3 + col
                                
                            color = cube_state[face_idx][sticker_idx]
                        else:
                            # This is an interior face, use the block color
                            color = "Block"
                            
                        # Calculate dot product with view direction to determine visibility
                        view_vector = (0, 0, 1)  # Simplified view vector after rotation
                        dot_product = nx * view_vector[0] + ny * view_vector[1] + nz * view_vector[2]
                        
                        # Store the face data for depth sorting
                        faces.append((proj_corners, color, avg_depth))
        
        # Sort faces by depth (from back to front)
        faces.sort(key=lambda f: f[2])
        
        # Draw faces from back to front
        for corners, color, _ in faces:
            if color == "Block":
                fill_color = self.block_color
                outline_color = self.block_color
                width = 0
            else:
                fill_color = self.colors[color]
                outline_color = "black"
                width = 1
                
            self.canvas.create_polygon(corners, fill=fill_color, outline=outline_color, width=width)

    def update_display(self):
        # Called to refresh the display
        self.canvas.update()

    def rotate_view(self, dx, dy):
        # Apply a consistent rotation factor to make movement smoother
        rotation_factor = 0.5
        
        # Rotate the view based on mouse drag with normalized factor
        self.rotation_y += dx * rotation_factor
        
        # Invert the x rotation to make it feel more natural
        self.rotation_x -= dy * rotation_factor  # Changed from += to -=
        
        # Ensure full 360-degree rotation is possible by using modulo
        self.rotation_y %= 360
        self.rotation_x %= 360