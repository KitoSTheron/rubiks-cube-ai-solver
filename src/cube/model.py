import copy

class Cube:
    def __init__(self):
        self.state = self.initialize_cube()
        self.solved_state_cache = copy.deepcopy(self.state)

    def initialize_cube(self):
        # Initialize the cube with a solved state
        return [
            ['W'] * 9,  # White (Top)
            ['R'] * 9,  # Red (Right)
            ['B'] * 9,  # Blue (Front)
            ['O'] * 9,  # Orange (Left)
            ['G'] * 9,  # Green (Back)
            ['Y'] * 9   # Yellow (Bottom)
        ]

    def rotate_face(self, face):
        # Map the face names to indices
        face_map = {
            'U': 0,  # Up/Top
            'R': 1,  # Right
            'F': 2,  # Front
            'L': 3,  # Left
            'B': 4,  # Back
            'D': 5   # Down/Bottom
        }
        
        if face not in face_map:
            return
            
        face_idx = face_map[face]
        
        # Rotate the face clockwise
        new_face = [
            self.state[face_idx][6], self.state[face_idx][3], self.state[face_idx][0],
            self.state[face_idx][7], self.state[face_idx][4], self.state[face_idx][1],
            self.state[face_idx][8], self.state[face_idx][5], self.state[face_idx][2]
        ]
        
        self.state[face_idx] = new_face
        
        # Now we need to update the adjacent faces
        self._rotate_adjacent_faces(face)

    def _rotate_adjacent_faces(self, face):
        # Define which stickers change when a face is rotated
        if face == 'U':  # Top face rotation
            # Save the front row
            front_row = self.state[2][:3]
            # Front <- Right
            self.state[2][:3] = self.state[1][:3]
            # Right <- Back
            self.state[1][:3] = self.state[4][:3]
            # Back <- Left
            self.state[4][:3] = self.state[3][:3]
            # Left <- Front (saved)
            self.state[3][:3] = front_row
            
        elif face == 'D':  # Bottom face rotation
            # Save the front bottom row
            front_row = self.state[2][6:9]
            # Front <- Left
            self.state[2][6:9] = self.state[3][6:9]
            # Left <- Back
            self.state[3][6:9] = self.state[4][6:9]
            # Back <- Right
            self.state[4][6:9] = self.state[1][6:9]
            # Right <- Front (saved)
            self.state[1][6:9] = front_row
            
        elif face == 'R':  # Right face rotation
            # This is more complex as we need to get specific columns
            # from adjacent faces
            # Save front column
            front_col = [self.state[2][2], self.state[2][5], self.state[2][8]]
            # Front <- Top
            self.state[2][2], self.state[2][5], self.state[2][8] = self.state[0][2], self.state[0][5], self.state[0][8]
            # Top <- Back (reversed)
            self.state[0][2], self.state[0][5], self.state[0][8] = self.state[4][6], self.state[4][3], self.state[4][0]
            # Back <- Bottom (reversed)
            self.state[4][6], self.state[4][3], self.state[4][0] = self.state[5][2], self.state[5][5], self.state[5][8]
            # Bottom <- Front
            self.state[5][2], self.state[5][5], self.state[5][8] = front_col
            
        # ... similar implementations for other faces (L, F, B)
        # For brevity, I'm only showing a few of the rotation implementations
        
    def get_state(self):
        # Return the current state of the cube
        return self.state
        
    def solved_state(self):
        # Return a solved state for comparison
        return self.solved_state_cache

    def is_solved(self):
        # Check if the cube is in a solved state
        for face in self.state:
            color = face[0]
            if not all(sticker == color for sticker in face):
                return False
        return True