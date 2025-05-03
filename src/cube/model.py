import threading

class Cube:
    def __init__(self):
        # Initialize a solved cube state
        self.state = [
            ['W'] * 9,  # Up (white)
            ['R'] * 9,  # Left (red)
            ['B'] * 9,  # Front (blue)
            ['O'] * 9,  # Right (orange)
            ['G'] * 9,  # Back (green)
            ['Y'] * 9   # Down (yellow)
        ]
        self.lock = threading.Lock()  # Lock for concurrency

    def get_state(self):
        return self.state

    def rotate_face(self, face, direction='clockwise'):
        """
        Rotate a face of the cube
        face: one of 'U', 'D', 'L', 'R', 'F', 'B'
        direction: 'clockwise', 'counterclockwise', or '180'
        """
        with self.lock:  # Ensure only one rotation happens at a time
            print(f"Rotating face {face} {direction}")
            print("State before rotation:")
            for i, face_state in enumerate(self.state):
                print(f"Face {i}: {face_state}")

            face_map = {'U': 0, 'R': 1, 'F': 2, 'L': 3, 'B': 4, 'D': 5}
            
            if face not in face_map:
                raise ValueError(f"Invalid face: {face}")
            
            face_idx = face_map[face]
            
            # For R, B, and D faces, invert the direction for face rotation
            actual_direction = direction
            if face in ['R', 'B', 'D']:
                if direction == 'clockwise':
                    actual_direction = 'counterclockwise'
                elif direction == 'counterclockwise':
                    actual_direction = 'clockwise'
            
            # 1. Rotate the face itself
            if actual_direction == 'clockwise':
                self._rotate_face_clockwise(face_idx)
            elif actual_direction == 'counterclockwise':
                self._rotate_face_counterclockwise(face_idx)
            elif actual_direction == '180':
                self._rotate_face_clockwise(face_idx)
                self._rotate_face_clockwise(face_idx)
            
            # 2. Rotate the adjacent edges (using same actual_direction)
            if face == 'U':
                self._rotate_U_edges(direction)
            elif face == 'D':
                self._rotate_D_edges(actual_direction)
            elif face == 'L':
                self._rotate_L_edges(direction)
            elif face == 'R':
                self._rotate_R_edges(actual_direction)
            elif face == 'F':
                self._rotate_F_edges(direction)
            elif face == 'B':
                self._rotate_B_edges(actual_direction)
            
            # Validate the cube state after rotation
            self._validate_state()
            self._validate_physical_constraints()  # Add this line to validate physical constraints

            print("State after rotation:")
            for i, face_state in enumerate(self.state):
                print(f"Face {i}: {face_state}")

    def _validate_state(self):
        """
        Validate the cube's state to ensure it is consistent and possible.
        """
        # Count the occurrences of each color
        color_counts = {}
        for face in self.state:
            for color in face:
                color_counts[color] = color_counts.get(color, 0) + 1
        
        # Each color should appear exactly 9 times
        for color, count in color_counts.items():
            if count != 9:
                raise ValueError(f"Invalid cube state: {color} appears {count} times (expected 9).")

    def _validate_physical_constraints(self):
        """
        Validate the cube's state to ensure it adheres to physical constraints.
        """
        # Define the expected pieces for a standard Rubik's Cube
        expected_edges = {
            frozenset(['W', 'R']), frozenset(['W', 'B']), frozenset(['W', 'O']), frozenset(['W', 'G']),
            frozenset(['Y', 'R']), frozenset(['Y', 'B']), frozenset(['Y', 'O']), frozenset(['Y', 'G']),
            frozenset(['R', 'B']), frozenset(['B', 'O']), frozenset(['O', 'G']), frozenset(['G', 'R'])
        }
        expected_corners = {
            frozenset(['W', 'R', 'B']), frozenset(['W', 'B', 'O']), frozenset(['W', 'O', 'G']), frozenset(['W', 'G', 'R']),
            frozenset(['Y', 'R', 'B']), frozenset(['Y', 'B', 'O']), frozenset(['Y', 'O', 'G']), frozenset(['Y', 'G', 'R'])
        }

        # Extract edges and corners from the current state
        edges = set()
        corners = set()

        # Define the positions of edges and corners on each face
        edge_mappings = [
            # (face1, index1, face2, index2)
            (0, 1, 1, 1),  # U-R
            (0, 5, 2, 1),  # U-F
            (0, 7, 3, 1),  # U-L
            (0, 3, 4, 1),  # U-B
            (5, 1, 1, 7),  # D-R
            (5, 5, 2, 7),  # D-F
            (5, 7, 3, 7),  # D-L
            (5, 3, 4, 7),  # D-B
            (2, 3, 3, 5),  # F-L
            (2, 5, 1, 3),  # F-R
            (4, 3, 1, 5),  # B-R
            (4, 5, 3, 3)   # B-L
        ]

        corner_mappings = [
            # (face1, index1, face2, index2, face3, index3)
            (0, 0, 3, 0, 4, 2),  # U-L-B
            (0, 2, 1, 0, 2, 2),  # U-R-F
            (0, 6, 3, 2, 2, 0),  # U-L-F
            (0, 8, 1, 2, 4, 0),  # U-R-B
            (5, 0, 3, 8, 4, 6),  # D-L-B
            (5, 2, 1, 8, 2, 6),  # D-R-F
            (5, 6, 3, 6, 2, 8),  # D-L-F
            (5, 8, 1, 6, 4, 8)   # D-R-B
        ]

        # Extract edges
        for face1, index1, face2, index2 in edge_mappings:
            edge = frozenset([self.state[face1][index1], self.state[face2][index2]])
            edges.add(edge)

        # Extract corners
        for face1, index1, face2, index2, face3, index3 in corner_mappings:
            corner = frozenset([self.state[face1][index1], self.state[face2][index2], self.state[face3][index3]])
            corners.add(corner)

        # Validate edges
        for edge in edges:
            if edge not in expected_edges:
                raise ValueError(f"Invalid edge piece: {edge}")

        # Validate corners
        for corner in corners:
            if corner not in expected_corners:
                raise ValueError(f"Invalid corner piece: {corner}")

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
    
    def _rotate_U_edges(self, direction):
        """Rotate edges adjacent to the U face"""
        F, R, B, L = 2, 1, 4, 3  # Face indices for Front, Right, Back, Left

        # Save the original values
        temp_F = self.state[F][0:3].copy()
        temp_R = self.state[R][0:3].copy()
        temp_B = self.state[B][0:3].copy()
        temp_L = self.state[L][0:3].copy()

        if direction == 'clockwise':
            self.state[F][0:3] = temp_L[0:3]
            self.state[R][0:3] = temp_F[0:3]
            self.state[B][0:3] = temp_R[0:3]
            self.state[L][0:3] = temp_B[0:3]
        elif direction == 'counterclockwise':
            self.state[F][0:3] = temp_R[0:3]
            self.state[R][0:3] = temp_B[0:3]
            self.state[B][0:3] = temp_L[0:3]
            self.state[L][0:3] = temp_F[0:3]
        elif direction == '180':
            self.state[F][0:3] = temp_B[0:3]
            self.state[R][0:3] = temp_L[0:3]
            self.state[B][0:3] = temp_F[0:3]
            self.state[L][0:3] = temp_R[0:3]

    def _rotate_D_edges(self, direction):
        """Rotate edges adjacent to the D face"""
        F, R, B, L = 2, 1, 4, 3  # Face indices for Front, Right, Back, Left

        # Save the original values (bottom row of each face)
        temp_F = self.state[F][6:9].copy()
        temp_R = self.state[R][6:9].copy()
        temp_B = self.state[B][6:9].copy()
        temp_L = self.state[L][6:9].copy()

        if direction == 'clockwise':
            # Rotate bottom rows clockwise
            self.state[F][6:9] = temp_L
            self.state[R][6:9] = temp_F
            self.state[B][6:9] = temp_R
            self.state[L][6:9] = temp_B
        elif direction == 'counterclockwise':
            # Rotate bottom rows counterclockwise
            self.state[F][6:9] = temp_R
            self.state[R][6:9] = temp_B
            self.state[B][6:9] = temp_L
            self.state[L][6:9] = temp_F
        elif direction == '180':
            # Rotate bottom rows 180 degrees
            self.state[F][6:9] = temp_B
            self.state[R][6:9] = temp_L
            self.state[B][6:9] = temp_F
            self.state[L][6:9] = temp_R
    
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