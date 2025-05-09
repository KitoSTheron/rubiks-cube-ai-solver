class DataModel:
    def __init__(self):

        self.top_face = 0 #Yellow
        self.front_face = 1 #Blue
        self.right_face = 2 #Red
        self.left_face = 3 #Orange
        self.back_face = 4 #Green
        self.bottom_face = 5 #White
        self.cube = [
            [['y0', 'y1', 'y2'], ['y3', 'y4', 'y5'], ['y6', 'y7', 'y8']],
            [['b0', 'b1', 'b2'], ['b3', 'b4', 'b5'], ['b6', 'b7', 'b8']],
            [['r0', 'r1', 'r2'], ['r3', 'r4', 'r5'], ['r6', 'r7', 'r8']],
            [['o0', 'o1', 'o2'], ['o3', 'o4', 'o5'], ['o6', 'o7', 'o8']],
            [['g0', 'g1', 'g2'], ['g3', 'g4', 'g5'], ['g6', 'g7', 'g8']],
            [['w0', 'w1', 'w2'], ['w3', 'w4', 'w5'], ['w6', 'w7', 'w8']]
        ]
        self.solved = [
            [['y0', 'y1', 'y2'], ['y3', 'y4', 'y5'], ['y6', 'y7', 'y8']],
            [['b0', 'b1', 'b2'], ['b3', 'b4', 'b5'], ['b6', 'b7', 'b8']],
            [['r0', 'r1', 'r2'], ['r3', 'r4', 'r5'], ['r6', 'r7', 'r8']],
            [['o0', 'o1', 'o2'], ['o3', 'o4', 'o5'], ['o6', 'o7', 'o8']],
            [['g0', 'g1', 'g2'], ['g3', 'g4', 'g5'], ['g6', 'g7', 'g8']],
            [['w0', 'w1', 'w2'], ['w3', 'w4', 'w5'], ['w6', 'w7', 'w8']]
        ]

        self.colors = ['yellow', 'blue', 'red', 'orange', 'green', 'white']
        self.color_map = {
            'yellow': 'top_face',
            'blue': 'front_face',
            'red': 'right_face',
            'orange': 'left_face',
            'green': 'back_face',
            'white': 'bottom_face'
        }
    
    def debug_cube_state(self, operation=""):
        """
        Print the current state of the cube for debugging
        
        Args:
            operation: String indicating which operation was just performed
        """
        print(f"\n--- Cube State after {operation} ---")
        
        face_names = {
            self.top_face: "TOP (Yellow)",
            self.front_face: "FRONT (Blue)",
            self.right_face: "RIGHT (Red)",
            self.left_face: "LEFT (Orange)",
            self.back_face: "BACK (Green)",
            self.bottom_face: "BOTTOM (White)"
        }
        
        for idx, face in face_names.items():
            print(f"{face}:")
            for row in self.cube[idx]:
                print(f"  {row}")
        print("-" * 40)

    def rotate_face(self, face, direction='clockwise'):
        """
        Rotate a face of the cube
        
        Args:
            face: One of 'U', 'D', 'L', 'R', 'F', 'B'
            direction: 'clockwise', 'counterclockwise', or 'double'
        """
        if face == 'R':
            if direction == 'clockwise':
                self.rotate_R()
            elif direction == 'counterclockwise':
                self.rotate_R_prime()
            elif direction == 'double':
                self.rotate_R2()
        elif face == 'L':
            if direction == 'clockwise':
                self.rotate_L()
            elif direction == 'counterclockwise':
                self.rotate_L_prime()
            elif direction == 'double':
                self.rotate_L2()
        elif face == 'U':
            if direction == 'clockwise':
                self.rotate_U()
            elif direction == 'counterclockwise':
                self.rotate_U_prime()
            elif direction == 'double':
                self.rotate_U2()
        elif face == 'D':
            if direction == 'clockwise':
                self.rotate_D()
            elif direction == 'counterclockwise':
                self.rotate_D_prime()
            elif direction == 'double':
                self.rotate_D2()
        elif face == 'F':
            if direction == 'clockwise':
                self.rotate_F()
            elif direction == 'counterclockwise':
                self.rotate_F_prime()
            elif direction == 'double':
                self.rotate_F2()
        elif face == 'B':
            if direction == 'clockwise':
                self.rotate_B()
            elif direction == 'counterclockwise':
                self.rotate_B_prime()
            elif direction == 'double':
                self.rotate_B2()
        else:
            raise ValueError(f"Invalid face '{face}' or direction '{direction}'")
        pass
        self.debug_cube_state()

    def rotate_R(self):
        """Rotate the Right face clockwise"""
        # Rotate the right face itself
        self.cube[self.right_face] = [list(row) for row in zip(*self.cube[self.right_face][::-1])]
        
        # Rotate the adjacent edges
        temp = [self.cube[self.top_face][i][2] for i in range(3)]
        for i in range(3):
            self.cube[self.top_face][i][2] = self.cube[self.front_face][i][2]
            self.cube[self.front_face][i][2] = self.cube[self.bottom_face][i][2]
            self.cube[self.bottom_face][i][2] = self.cube[self.back_face][2 - i][0]
            self.cube[self.back_face][2 - i][0] = temp[i]

    def rotate_R_prime(self):
        """Rotate the Right face counterclockwise"""
        # Rotate the right face itself
        self.cube[self.right_face] = [list(row) for row in zip(*self.cube[self.right_face])][::-1]
        
        # Rotate the adjacent edges
        temp = [self.cube[self.top_face][i][2] for i in range(3)]
        for i in range(3):
            self.cube[self.top_face][i][2] = self.cube[self.back_face][2 - i][0]
            self.cube[self.back_face][2 - i][0] = self.cube[self.bottom_face][i][2]
            self.cube[self.bottom_face][i][2] = self.cube[self.front_face][i][2]
            self.cube[self.front_face][i][2] = temp[i]

    def rotate_R2(self):
        """Rotate the Right face twice (180 degrees)"""
        self.rotate_R()
        self.rotate_R()

    def rotate_L(self):
        """Rotate the Left face clockwise"""
        # Rotate the left face itself
        self.cube[self.left_face] = [list(row) for row in zip(*self.cube[self.left_face])][::-1]
        
        # Rotate the adjacent edges
        temp = [self.cube[self.top_face][i][0] for i in range(3)]
        for i in range(3):
            self.cube[self.top_face][i][0] = self.cube[self.front_face][i][0]
            self.cube[self.front_face][i][0] = self.cube[self.bottom_face][i][0]
            self.cube[self.bottom_face][i][0] = self.cube[self.back_face][2 - i][2]
            self.cube[self.back_face][2 - i][2] = temp[i]

    def rotate_L_prime(self):
        """Rotate the Left face counterclockwise"""
        # Rotate the left face itself
        self.cube[self.left_face] = [list(row) for row in zip(*self.cube[self.left_face][::-1])]
        
        # Rotate the adjacent edges
        temp = [self.cube[self.top_face][i][0] for i in range(3)]
        for i in range(3):
            self.cube[self.top_face][i][0] = self.cube[self.back_face][2 - i][2]
            self.cube[self.back_face][2 - i][2] = self.cube[self.bottom_face][i][0]
            self.cube[self.bottom_face][i][0] = self.cube[self.front_face][i][0]
            self.cube[self.front_face][i][0] = temp[i]
        

    def rotate_L2(self):
        """Rotate the Left face twice (180 degrees)"""
        self.rotate_L()
        self.rotate_L()

    def rotate_U(self):
        """Rotate the Upper face clockwise"""
        # Rotate the top face itself
        self.cube[self.top_face] = [list(row) for row in zip(*self.cube[self.top_face][::-1])]
        
        # Rotate the adjacent edges
        temp = self.cube[self.front_face][0][:]
        self.cube[self.front_face][0] = self.cube[self.right_face][0]
        self.cube[self.right_face][0] = self.cube[self.back_face][0]
        self.cube[self.back_face][0] = self.cube[self.left_face][0]
        self.cube[self.left_face][0] = temp

    def rotate_U_prime(self):
        """Rotate the Upper face counterclockwise"""
        # Rotate the top face itself
        self.cube[self.top_face] = [list(row) for row in zip(*self.cube[self.top_face])][::-1]
        
        # Rotate the adjacent edges
        temp = self.cube[self.front_face][0][:]
        self.cube[self.front_face][0] = self.cube[self.left_face][0]
        self.cube[self.left_face][0] = self.cube[self.back_face][0]
        self.cube[self.back_face][0] = self.cube[self.right_face][0]
        self.cube[self.right_face][0] = temp

    def rotate_U2(self):
        """Rotate the Upper face twice (180 degrees)"""
        self.rotate_U()
        self.rotate_U()

    def rotate_D(self):
        """Rotate the Down face clockwise"""
        # Rotate the bottom face itself
        self.cube[self.bottom_face] = [list(row) for row in zip(*self.cube[self.bottom_face][::-1])]
        
        # Rotate the adjacent edges
        temp = self.cube[self.front_face][2][:]
        self.cube[self.front_face][2] = self.cube[self.left_face][2]
        self.cube[self.left_face][2] = self.cube[self.back_face][2]
        self.cube[self.back_face][2] = self.cube[self.right_face][2]
        self.cube[self.right_face][2] = temp

    def rotate_D_prime(self):
        """Rotate the Down face counterclockwise"""
        # Rotate the bottom face itself
        self.cube[self.bottom_face] = [list(row) for row in zip(*self.cube[self.bottom_face])][::-1]
        
        # Rotate the adjacent edges
        temp = self.cube[self.front_face][2][:]
        self.cube[self.front_face][2] = self.cube[self.right_face][2]
        self.cube[self.right_face][2] = self.cube[self.back_face][2]
        self.cube[self.back_face][2] = self.cube[self.left_face][2]
        self.cube[self.left_face][2] = temp

    def rotate_D2(self):
        """Rotate the Down face twice (180 degrees)"""
        self.rotate_D()
        self.rotate_D()

    def rotate_F(self):
        """Rotate the Front face clockwise"""
        # Rotate the front face itself
        self.cube[self.front_face] = [list(row) for row in zip(*self.cube[self.front_face][::-1])]
        
        # Rotate the adjacent edges
        temp = self.cube[self.top_face][2][:]
        for i in range(3):
            self.cube[self.top_face][2][i] = self.cube[self.left_face][2 - i][2]
            self.cube[self.left_face][2 - i][2] = self.cube[self.bottom_face][0][2 - i]
            self.cube[self.bottom_face][0][2 - i] = self.cube[self.right_face][i][0]
            self.cube[self.right_face][i][0] = temp[i]

    def rotate_F_prime(self):
        """Rotate the Front face counterclockwise"""
        # Rotate the front face itself
        self.cube[self.front_face] = [list(row) for row in zip(*self.cube[self.front_face])][::-1]
        
        # Rotate the adjacent edges
        temp = self.cube[self.top_face][2][:]
        for i in range(3):
            self.cube[self.top_face][2][i] = self.cube[self.right_face][i][0]
            self.cube[self.right_face][i][0] = self.cube[self.bottom_face][0][2 - i]
            self.cube[self.bottom_face][0][2 - i] = self.cube[self.left_face][2 - i][2]
            self.cube[self.left_face][2 - i][2] = temp[i]

    def rotate_F2(self):
        """Rotate the Front face twice (180 degrees)"""
        self.rotate_F()
        self.rotate_F()

    def rotate_B(self):
        """Rotate the Back face clockwise"""
        # Rotate the back face itself
        self.cube[self.back_face] = [list(row) for row in zip(*self.cube[self.back_face])][::-1]
        
        # Rotate the adjacent edges
        temp = self.cube[self.top_face][0][:]
        for i in range(3):
            self.cube[self.top_face][0][i] = self.cube[self.left_face][2 - i][0]
            self.cube[self.left_face][2 - i][0] = self.cube[self.bottom_face][2][2 - i]
            self.cube[self.bottom_face][2][2 - i] = self.cube[self.right_face][i][2]
            self.cube[self.right_face][i][2] = temp[i]

    def rotate_B_prime(self):
        """Rotate the Back face counterclockwise"""
        # Rotate the back face itself
        self.cube[self.back_face] = [list(row) for row in zip(*self.cube[self.back_face][::-1])]
        
        # Rotate the adjacent edges
        temp = self.cube[self.top_face][0][:]
        for i in range(3):
            self.cube[self.top_face][0][i] = self.cube[self.right_face][i][2]
            self.cube[self.right_face][i][2] = self.cube[self.bottom_face][2][2 - i]
            self.cube[self.bottom_face][2][2 - i] = self.cube[self.left_face][2 - i][0]
            self.cube[self.left_face][2 - i][0] = temp[i]

    def rotate_B2(self):
        """Rotate the Back face twice (180 degrees)"""
        self.rotate_B()
        self.rotate_B()

    

