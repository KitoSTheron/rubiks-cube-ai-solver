class Cube:
    def __init__(self):
        # Initialize a solved cube state
        # Each face is represented as a list of 9 colors (0-8, starting from top-left)
        # The order of faces is: [U, R, F, L, B, D]
        self.state = [
            ['W'] * 9,  # Up (white)
            ['R'] * 9,  # Left (red)
            ['B'] * 9,  # Front (blue)
            ['O'] * 9,  # Right (orange)
            ['G'] * 9,  # Back (green)
            ['Y'] * 9   # Down (yellow)
        ]
    
    def get_state(self):
        return self.state
    
    def rotate_face(self, face, direction='clockwise'):
        """
        Rotate a face of the cube
        face: one of 'U', 'D', 'L', 'R', 'F', 'B'
        direction: 'clockwise', 'counterclockwise', or '180'
        """
        face_map = {'U': 0, 'R': 1, 'F': 2, 'L': 3, 'B': 4, 'D': 5}
        
        if face not in face_map:
            raise ValueError(f"Invalid face: {face}")
        
        face_idx = face_map[face]
        
        # 1. Rotate the face itself
        if direction == 'clockwise':
            self._rotate_face_clockwise(face_idx)
        elif direction == 'counterclockwise':
            self._rotate_face_counterclockwise(face_idx)
        elif direction == '180':
            self._rotate_face_clockwise(face_idx)
            self._rotate_face_clockwise(face_idx)
        
        # 2. Rotate the adjacent edges
        self._rotate_adjacent_edges(face, direction)
        
    def _rotate_face_clockwise(self, face_idx):
        """Rotate the stickers on a single face clockwise"""
        face = self.state[face_idx]
        # Save the original state
        temp = face.copy()
        # Apply rotation
        face[0] = temp[6]
        face[1] = temp[3]
        face[2] = temp[0]
        face[3] = temp[7]
        face[5] = temp[1]
        face[6] = temp[8]
        face[7] = temp[5]
        face[8] = temp[2]
        # Note: Center piece (index 4) doesn't move
    
    def _rotate_face_counterclockwise(self, face_idx):
        """Rotate the stickers on a single face counterclockwise"""
        # Apply clockwise rotation three times
        for _ in range(3):
            self._rotate_face_clockwise(face_idx)
    
    def _rotate_adjacent_edges(self, face, direction):
        """Rotate the edges adjacent to the given face"""
        # Define the adjacent edges for each face
        if face == 'U':
            self._rotate_U_edges(direction)
        elif face == 'D':
            self._rotate_D_edges(direction)
        elif face == 'L':
            self._rotate_L_edges(direction)
        elif face == 'R':
            self._rotate_R_edges(direction)
        elif face == 'F':
            self._rotate_F_edges(direction)
        elif face == 'B':
            self._rotate_B_edges(direction)
    
    def _rotate_U_edges(self, direction):
        """Rotate edges adjacent to the U face"""
        # The affected edges are the top rows of F, R, B, L
        F, R, B, L = 2, 1, 4, 3
        
        # Save the original values
        temp_F = self.state[F][0:3].copy()  # Top row of Front
        temp_R = self.state[R][0:3].copy()  # Top row of Right
        temp_B = self.state[B][0:3].copy()  # Top row of Back
        temp_L = self.state[L][0:3].copy()  # Top row of Left
        
        if direction == 'clockwise':
            # Front gets Left's top
            self.state[F][0:3] = temp_L[0:3]
            # Right gets Front's top
            self.state[R][0:3] = temp_F[0:3]
            # Back gets Right's top
            self.state[B][0:3] = temp_R[0:3]
            # Left gets Back's top
            self.state[L][0:3] = temp_B[0:3]
        elif direction == 'counterclockwise':
            # Front gets Right's top
            self.state[F][0:3] = temp_R[0:3]
            # Right gets Back's top
            self.state[R][0:3] = temp_B[0:3]
            # Back gets Left's top
            self.state[B][0:3] = temp_L[0:3]
            # Left gets Front's top
            self.state[L][0:3] = temp_F[0:3]
        elif direction == '180':
            # Front gets Back's top
            self.state[F][0:3] = temp_B[0:3]
            # Right gets Left's top
            self.state[R][0:3] = temp_L[0:3]
            # Back gets Front's top
            self.state[B][0:3] = temp_F[0:3]
            # Left gets Right's top
            self.state[L][0:3] = temp_R[0:3]
    
    def _rotate_D_edges(self, direction):
        """Rotate edges adjacent to the D face"""
        # The affected edges are the bottom rows of F, R, B, L
        F, R, B, L = 2, 1, 4, 3
        
        # Save the original values - explicitly accessing indices 6, 7, 8
        temp_F = [self.state[F][6], self.state[F][7], self.state[F][8]]  # Bottom row of Front
        temp_R = [self.state[R][6], self.state[R][7], self.state[R][8]]  # Bottom row of Right
        temp_B = [self.state[B][6], self.state[B][7], self.state[B][8]]  # Bottom row of Back
        temp_L = [self.state[L][6], self.state[L][7], self.state[L][8]]  # Bottom row of Left
        
        if direction == 'clockwise':
            # Front gets Right's bottom
            self.state[F][6], self.state[F][7], self.state[F][8] = temp_R[0], temp_R[1], temp_R[2]
            # Right gets Back's bottom
            self.state[R][6], self.state[R][7], self.state[R][8] = temp_B[0], temp_B[1], temp_B[2]
            # Back gets Left's bottom
            self.state[B][6], self.state[B][7], self.state[B][8] = temp_L[0], temp_L[1], temp_L[2]
            # Left gets Front's bottom
            self.state[L][6], self.state[L][7], self.state[L][8] = temp_F[0], temp_F[1], temp_F[2]
        elif direction == 'counterclockwise':
            # Front gets Left's bottom
            self.state[F][6], self.state[F][7], self.state[F][8] = temp_L[0], temp_L[1], temp_L[2]
            # Right gets Front's bottom
            self.state[R][6], self.state[R][7], self.state[R][8] = temp_F[0], temp_F[1], temp_F[2]
            # Back gets Right's bottom
            self.state[B][6], self.state[B][7], self.state[B][8] = temp_R[0], temp_R[1], temp_R[2]
            # Left gets Back's bottom
            self.state[L][6], self.state[L][7], self.state[L][8] = temp_B[0], temp_B[1], temp_B[2]
        elif direction == '180':
            # Instead of calling rotate_D_edges twice, do it directly for better performance
            # Front gets Back's bottom
            self.state[F][6], self.state[F][7], self.state[F][8] = temp_B[0], temp_B[1], temp_B[2]
            # Right gets Left's bottom
            self.state[R][6], self.state[R][7], self.state[R][8] = temp_L[0], temp_L[1], temp_L[2]
            # Back gets Front's bottom
            self.state[B][6], self.state[B][7], self.state[B][8] = temp_F[0], temp_F[1], temp_F[2]
            # Left gets Right's bottom
            self.state[L][6], self.state[L][7], self.state[L][8] = temp_R[0], temp_R[1], temp_R[2]
    
    def _rotate_F_edges(self, direction):
        """Rotate edges adjacent to the F face"""
        # The affected edges are:
        # - Bottom row of U
        # - Left column of R
        # - Top row of D
        # - Right column of L
        U, R, D, L = 0, 1, 5, 3
        
        # Save the original values
        temp_U = [self.state[U][6], self.state[U][7], self.state[U][8]]  # Bottom row of Up
        temp_R = [self.state[R][0], self.state[R][3], self.state[R][6]]  # Left column of Right
        temp_D = [self.state[D][0], self.state[D][1], self.state[D][2]]  # Top row of Down
        temp_L = [self.state[L][2], self.state[L][5], self.state[L][8]]  # Right column of Left
        
        if direction == 'clockwise':
            # Up's bottom row gets Left's right column (reversed)
            self.state[U][6], self.state[U][7], self.state[U][8] = temp_L[2], temp_L[1], temp_L[0]
            # Right's left column gets Up's bottom row
            self.state[R][0], self.state[R][3], self.state[R][6] = temp_U[0], temp_U[1], temp_U[2]
            # Down's top row gets Right's left column (reversed)
            self.state[D][0], self.state[D][1], self.state[D][2] = temp_R[2], temp_R[1], temp_R[0]
            # Left's right column gets Down's top row
            self.state[L][2], self.state[L][5], self.state[L][8] = temp_D[0], temp_D[1], temp_D[2]
        elif direction == 'counterclockwise':
            # Up's bottom row gets Right's left column
            self.state[U][6], self.state[U][7], self.state[U][8] = temp_R[0], temp_R[1], temp_R[2]
            # Right's left column gets Down's top row (reversed)
            self.state[R][0], self.state[R][3], self.state[R][6] = temp_D[2], temp_D[1], temp_D[0]
            # Down's top row gets Left's right column
            self.state[D][0], self.state[D][1], self.state[D][2] = temp_L[0], temp_L[1], temp_L[2]
            # Left's right column gets Up's bottom row (reversed)
            self.state[L][2], self.state[L][5], self.state[L][8] = temp_U[2], temp_U[1], temp_U[0]
        elif direction == '180':
            # Apply twice
            self._rotate_F_edges('clockwise')
            self._rotate_F_edges('clockwise')
    
    def _rotate_B_edges(self, direction):
        """Rotate edges adjacent to the B face"""
        # The affected edges are:
        # - Top row of U
        # - Right column of R
        # - Bottom row of D
        # - Left column of L
        U, R, D, L = 0, 1, 5, 3
        
        # Save the original values
        temp_U = [self.state[U][0], self.state[U][1], self.state[U][2]]  # Top row of Up
        temp_R = [self.state[R][2], self.state[R][5], self.state[R][8]]  # Right column of Right
        temp_D = [self.state[D][6], self.state[D][7], self.state[D][8]]  # Bottom row of Down
        temp_L = [self.state[L][0], self.state[L][3], self.state[L][6]]  # Left column of Left
        
        if direction == 'clockwise':
            # Up's top row gets Right's right column (reversed)
            self.state[U][0], self.state[U][1], self.state[U][2] = temp_R[2], temp_R[1], temp_R[0]
            
            # Right's right column gets Down's bottom row
            self.state[R][2], self.state[R][5], self.state[R][8] = temp_D[0], temp_D[1], temp_D[2]
            
            # Down's bottom row gets Left's left column (reversed)
            self.state[D][6], self.state[D][7], self.state[D][8] = temp_L[2], temp_L[1], temp_L[0]
            
            # Left's left column gets Up's top row
            self.state[L][0], self.state[L][3], self.state[L][6] = temp_U[0], temp_U[1], temp_U[2]
            
        elif direction == 'counterclockwise':
            # Up's top row gets Left's left column
            self.state[U][0], self.state[U][1], self.state[U][2] = temp_L[0], temp_L[1], temp_L[2]
            
            # Right's right column gets Up's top row (reversed)
            self.state[R][2], self.state[R][5], self.state[R][8] = temp_U[2], temp_U[1], temp_U[0]
            
            # Down's bottom row gets Right's right column
            self.state[D][6], self.state[D][7], self.state[D][8] = temp_R[0], temp_R[1], temp_R[2]
            
            # Left's left column gets Down's bottom row (reversed)
            self.state[L][0], self.state[L][3], self.state[L][6] = temp_D[2], temp_D[1], temp_D[0]
            
        elif direction == '180':
            # Implement 180-degree rotation directly for better efficiency
            # Up's top row gets Down's bottom row (reversed)
            self.state[U][0], self.state[U][1], self.state[U][2] = temp_D[2], temp_D[1], temp_D[0]
            
            # Right's right column gets Left's left column (reversed)
            self.state[R][2], self.state[R][5], self.state[R][8] = temp_L[2], temp_L[1], temp_L[0]
            
            # Down's bottom row gets Up's top row (reversed)
            self.state[D][6], self.state[D][7], self.state[D][8] = temp_U[2], temp_U[1], temp_U[0]
            
            # Left's left column gets Right's right column (reversed)
            self.state[L][0], self.state[L][3], self.state[L][6] = temp_R[2], temp_R[1], temp_R[0]
    
    def _rotate_R_edges(self, direction):
        """Rotate edges adjacent to the R face"""
        # The affected edges are:
        # - Right column of U
        # - Right column of F
        # - Right column of D
        # - Left column of B
        U, F, D, B = 0, 2, 5, 4
        
        # Save the original values
        temp_U = [self.state[U][2], self.state[U][5], self.state[U][8]]  # Right column of Up
        temp_F = [self.state[F][2], self.state[F][5], self.state[F][8]]  # Right column of Front
        temp_D = [self.state[D][2], self.state[D][5], self.state[D][8]]  # Right column of Down
        temp_B = [self.state[B][0], self.state[B][3], self.state[B][6]]  # Left column of Back (reversed)
        
        if direction == 'clockwise':
            # Up's right column gets Back's left column (reversed)
            self.state[U][2], self.state[U][5], self.state[U][8] = temp_B[2], temp_B[1], temp_B[0]
            # Front's right column gets Up's right column
            self.state[F][2], self.state[F][5], self.state[F][8] = temp_U[0], temp_U[1], temp_U[2]
            # Down's right column gets Front's right column
            self.state[D][2], self.state[D][5], self.state[D][8] = temp_F[0], temp_F[1], temp_F[2]
            # Back's left column gets Down's right column (reversed)
            self.state[B][0], self.state[B][3], self.state[B][6] = temp_D[2], temp_D[1], temp_D[0]
        elif direction == 'counterclockwise':
            # Up's right column gets Front's right column
            self.state[U][2], self.state[U][5], self.state[U][8] = temp_F[0], temp_F[1], temp_F[2]
            # Front's right column gets Down's right column
            self.state[F][2], self.state[F][5], self.state[F][8] = temp_D[0], temp_D[1], temp_D[2]
            # Down's right column gets Back's left column (reversed)
            self.state[D][2], self.state[D][5], self.state[D][8] = temp_B[2], temp_B[1], temp_B[0]
            # Back's left column gets Up's right column (reversed)
            self.state[B][0], self.state[B][3], self.state[B][6] = temp_U[2], temp_U[1], temp_U[0]
        elif direction == '180':
            # Apply twice
            self._rotate_R_edges('clockwise')
            self._rotate_R_edges('clockwise')
    
    def _rotate_L_edges(self, direction):
        """Rotate edges adjacent to the L face"""
        # The affected edges are:
        # - Left column of U
        # - Left column of F
        # - Left column of D
        # - Right column of B
        U, F, D, B = 0, 2, 5, 4
        
        # Save the original values
        temp_U = [self.state[U][0], self.state[U][3], self.state[U][6]]  # Left column of Up
        temp_F = [self.state[F][0], self.state[F][3], self.state[F][6]]  # Left column of Front
        temp_D = [self.state[D][0], self.state[D][3], self.state[D][6]]  # Left column of Down
        temp_B = [self.state[B][2], self.state[B][5], self.state[B][8]]  # Right column of Back (reversed)
        
        if direction == 'clockwise':
            # Up's left column gets Front's left column
            self.state[U][0], self.state[U][3], self.state[U][6] = temp_F[0], temp_F[1], temp_F[2]
            # Front's left column gets Down's left column
            self.state[F][0], self.state[F][3], self.state[F][6] = temp_D[0], temp_D[1], temp_D[2]
            # Down's left column gets Back's right column (reversed)
            self.state[D][0], self.state[D][3], self.state[D][6] = temp_B[2], temp_B[1], temp_B[0]
            # Back's right column gets Up's left column (reversed)
            self.state[B][2], self.state[B][5], self.state[B][8] = temp_U[2], temp_U[1], temp_U[0]
        elif direction == 'counterclockwise':
            # Up's left column gets Back's right column (reversed)
            self.state[U][0], self.state[U][3], self.state[U][6] = temp_B[2], temp_B[1], temp_B[0]
            # Front's left column gets Up's left column
            self.state[F][0], self.state[F][3], self.state[F][6] = temp_U[0], temp_U[1], temp_U[2]
            # Down's left column gets Front's left column
            self.state[D][0], self.state[D][3], self.state[D][6] = temp_F[0], temp_F[1], temp_F[2]
            # Back's right column gets Down's left column (reversed)
            self.state[B][2], self.state[B][5], self.state[B][8] = temp_D[2], temp_D[1], temp_D[0]
        elif direction == '180':
            # Apply twice
            self._rotate_L_edges('clockwise')
            self._rotate_L_edges('clockwise')