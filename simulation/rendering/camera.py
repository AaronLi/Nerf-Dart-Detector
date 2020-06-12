import numpy as np

class Camera:
    UP = np.array((0, 1, 0, 0))

    def __init__(self, pos = None, gaze = None) -> None:
        if pos is None:
            self.pos = np.zeros(4)
            self.pos[3] = 1
        else:
            self.pos = pos

        if gaze is None:
            self.gaze = np.zeros(4)
            self.gaze[0] = 1
            self.gaze[3] = 1
        else:
            self.gaze = gaze

        self.u = np.zeros(4) # u is right in camera coords

        self.v = np.zeros(4) # v is "up" in camera coords

        self.n = np.zeros(4) # n points at the target from the position

        self.calculate_view_vectors()


    def calculate_view_vectors(self):
        n = (self.gaze - self.pos)
        self.n = (n / np.linalg.norm(n))

        u = np.cross(n[:3], Camera.UP[:3])
        self.u = np.hstack((u / np.linalg.norm(u), 0))

        v = np.cross(u[:3], n[:3])
        self.v = np.hstack((v / np.linalg.norm(v), 0))