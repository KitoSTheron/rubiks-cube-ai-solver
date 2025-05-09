class DataModel:
    def __init__(self):

        self.top_face = 0 #Yellow
        self.front_face = 1 #Blue
        self.right_face = 2 #Red
        self.left_face = 3 #Orange
        self.back_face = 4 #Green
        self.bottom_face = 5 #White
        self.faces = [
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

    def rotate_R(self):
        """Rotate the Right face clockwise"""
        self.rotate_face('R', 'clockwise')

    def rotate_R_prime(self):
        """Rotate the Right face counterclockwise"""
        self.rotate_face('R', 'counterclockwise')

    def rotate_R2(self):
        """Rotate the Right face twice (180 degrees)"""
        self.rotate_face('R', 'double')

    def rotate_L(self):
        """Rotate the Left face clockwise"""
        self.rotate_face('L', 'clockwise')

    def rotate_L_prime(self):
        """Rotate the Left face counterclockwise"""
        self.rotate_face('L', 'counterclockwise')

    def rotate_L2(self):
        """Rotate the Left face twice (180 degrees)"""
        self.rotate_face('L', 'double')

    def rotate_U(self):
        """Rotate the Upper face clockwise"""
        self.rotate_face('U', 'clockwise')

    def rotate_U_prime(self):
        """Rotate the Upper face counterclockwise"""
        self.rotate_face('U', 'counterclockwise')

    def rotate_U2(self):
        """Rotate the Upper face twice (180 degrees)"""
        self.rotate_face('U', 'double')

    def rotate_D(self):
        """Rotate the Down face clockwise"""
        self.rotate_face('D', 'clockwise')

    def rotate_D_prime(self):
        """Rotate the Down face counterclockwise"""
        self.rotate_face('D', 'counterclockwise')

    def rotate_D2(self):
        """Rotate the Down face twice (180 degrees)"""
        self.rotate_face('D', 'double')

    def rotate_F(self):
        """Rotate the Front face clockwise"""
        self.rotate_face('F', 'clockwise')

    def rotate_F_prime(self):
        """Rotate the Front face counterclockwise"""
        self.rotate_face('F', 'counterclockwise')

    def rotate_F2(self):
        """Rotate the Front face twice (180 degrees)"""
        self.rotate_face('F', 'double')

    def rotate_B(self):
        """Rotate the Back face clockwise"""
        self.rotate_face('B', 'clockwise')

    def rotate_B_prime(self):
        """Rotate the Back face counterclockwise"""
        self.rotate_face('B', 'counterclockwise')

    def rotate_B2(self):
        """Rotate the Back face twice (180 degrees)"""
        self.rotate_face('B', 'double')

