import numpy as np
from simulation.rendering import transform

class GenericShape:
    X = np.array((1, 0, 0, 0))
    Y = np.array((0, 1, 0, 0))
    Z = np.array((0, 0, 1, 0))
    def __init__(self, colour = (255, 255, 255), transform = np.identity(4, dtype=float), pos = None):
        self.transform = transform
        if pos is None:
            self.pos = np.zeros(4)
            self.pos[3] = 1
        else:
            self.pos = pos
        self.faces = np.zeros((1, 1, 4), dtype=float)
        self.colour = colour

    def get_world_faces(self):
        out = np.ndarray(self.faces.shape, dtype=float)
        for i,face in enumerate(self.faces):
            for j, point in enumerate(face):
                out[i][j] = np.dot(self.transform, point) + self.pos
                # I wish pos could be a vector so I wouldn't have to do this, but using a translation matrix requires it
                # be a point
                out[i][j][3] = 1

        return out
    def translate(self, axis_vector: np.ndarray):
        translation_matrix = transform.translate(axis_vector)

        self.pos = np.dot(translation_matrix, self.pos)

    def translate_x(self, amount):
        self.translate(np.array((amount, 0, 0, 0)))

    def translate_y(self, amount):
        self.translate(np.array((0, amount, 0, 0)))

    def translate_z(self, amount):
        self.translate(np.array((0, 0, amount, 0)))

    def scale(self, scale_vector: np.ndarray):
        scale_matrix = transform.scale(scale_vector)

        self.transform = np.dot(scale_matrix, self.transform)

    def scale_x(self, amount):
        self.scale(np.array((amount, 1, 1, 1)))

    def scale_y(self, amount):
        self.scale(np.array((1, amount, 1, 1)))

    def scale_z(self, amount):
        self.scale(np.array((1, 1, amount, 1)))

    def rotate(self, axis_vector, angle):
        homogeneous_rotation_matrix = transform.rotate(axis_vector, angle)

        self.transform = np.dot(homogeneous_rotation_matrix, self.transform)

    def rotate_x(self, angle):
        self.rotate(GenericShape.X, angle)

    def rotate_y(self, angle):
        self.rotate(GenericShape.Y, angle)

    def rotate_z(self, angle):
        self.rotate(GenericShape.Z, angle)