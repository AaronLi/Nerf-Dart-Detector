from simulation.rendering.shape.generic import GenericShape
import numpy as np


class Cylinder(GenericShape):
    def __init__(self, diameter, height, faces=20, transform=np.identity(4, dtype=float)):
        super().__init__(transform)

        self.scale_x(diameter)
        self.scale_y(height)
        self.scale_z(diameter)

        radians_per_face = 2 * np.pi / faces

        self.faces = np.ndarray((faces, 4, 4), dtype=float)

        for i in range(faces):
            face = np.ndarray((4, 4), dtype=float)
            # create a 4 point polygon for each face of the cylinder (no end caps)
            face[0] = (np.sin(i * radians_per_face), 0, np.cos(i * radians_per_face), 1)
            face[1] = (np.sin((i + 1) * radians_per_face), 0, np.cos((i + 1) * radians_per_face), 1)
            face[2] = (np.sin((i + 1) * radians_per_face), 1, np.cos((i + 1) * radians_per_face), 1)
            face[3] = (np.sin(i * radians_per_face), 1, np.cos(i * radians_per_face), 1)
            self.faces[i] = face
