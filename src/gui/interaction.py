class Interaction:
    def __init__(self, cube_renderer, cube):
        self.cube_renderer = cube_renderer
        self.cube = cube  # Store direct reference to the cube
        self.last_x = 0
        self.last_y = 0
        self.dragging = False
        
    def handle_click(self, event):
        self.last_x = event.x
        self.last_y = event.y
        self.dragging = True
        
    def handle_drag(self, event):
        if not self.dragging:
            return
            
        # Calculate the delta movement
        dx = event.x - self.last_x
        dy = event.y - self.last_y
        

        
        # Update the cube rotation
        self.cube_renderer.rotate_view(dx, dy)
        
        # Redraw the cube using the direct cube reference
        self.cube_renderer.draw_cube(self.cube.get_state())
        
        # Update the last position
        self.last_x = event.x
        self.last_y = event.y
        
    def handle_release(self, event):
        self.dragging = False
        
    def update_cube_state(self):
        # Method to update the cube state based on user interactions
        # This would be called when the user makes a move on the cube
        self.cube_renderer.draw_cube(self.cube.get_state())