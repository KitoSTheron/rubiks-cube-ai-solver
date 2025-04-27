class Interaction:
    def __init__(self, cube_renderer, cube):
        self.cube_renderer = cube_renderer
        self.cube = cube  # Store direct reference to the cube
        self.last_x = 0
        self.last_y = 0
        self.dragging = False
        self.drag_distance = 0
        self.start_x = 0
        self.start_y = 0
        
    def handle_click(self, event):
        self.last_x = event.x
        self.last_y = event.y
        self.start_x = event.x
        self.start_y = event.y
        self.drag_distance = 0
        self.dragging = True
        
    def handle_drag(self, event):
        if not self.dragging:
            return
            
        # Calculate the delta movement
        dx = event.x - self.last_x
        dy = event.y - self.last_y
        
        # Accumulate drag distance
        self.drag_distance += (dx**2 + dy**2)**0.5

        # Update the cube rotation
        self.cube_renderer.rotate_view(dx, dy)
        
        # Redraw the cube using the direct cube reference
        self.cube_renderer.draw_cube(self.cube.get_state())
        
        # Update the last position
        self.last_x = event.x
        self.last_y = event.y
        
    def handle_release(self, event):
        self.dragging = False
        
        # If the drag distance is small, treat it as a click on a face
        if self.drag_distance < 5:
            self.handle_face_click(self.start_x, self.start_y)
            
    def handle_face_click(self, x, y):
        """Determine which face was clicked and rotate it"""
        # Get the face and direction from renderer based on screen coordinates
        face_info = self.cube_renderer.get_face_at_position(x, y)
        if face_info:
            face, clockwise = face_info
            # Rotate the face
            if clockwise:
                self.cube.rotate_face(face, 'clockwise')
            else:
                self.cube.rotate_face(face, 'counterclockwise')
        
            # Update the display
            self.update_cube_state()
        
    def update_cube_state(self):
        # Method to update the cube state based on user interactions
        # This would be called when the user makes a move on the cube
        self.cube_renderer.draw_cube(self.cube.get_state())
        
    def handle_swipe(self, event, direction):
        """Handle swipe gestures for face rotation"""
        face_info = self.cube_renderer.get_face_for_swipe(event.x, event.y, direction)
        if face_info:
            face, clockwise = face_info
            if clockwise:
                self.cube.rotate_face(face, 'clockwise')
            else:
                self.cube.rotate_face(face, 'counterclockwise')
            self.update_cube_state()