import numpy as np
from typing import List
from simulation.rendering.shape import *
from collections import namedtuple
from simulation.rendering import light_source

RenderedPolygon = namedtuple("RenderedPolygon", ("face", "colour"))


class Camera:
    """
    A virtual camera for use in projecting 3d scenes onto a 2d plane. +X is camera forward. +Y is Screen Y / Camera Up.
    +Z is Screen X / Camera Right
    """
    UP = np.array((0, 1, 0, 0))

    SWAP_YZ = np.array(
        (
            (1, 0, 0, 0),
            (0, 0, 1, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 1)
        )
    )

    def __init__(self, h_resolution, v_resolution, fov, near_plane=2, far_plane=30, pos=None, gaze=None) -> None:

        self.fov = fov
        self.v_resolution = v_resolution
        self.h_resolution = h_resolution

        self.aspect_ratio = self.h_resolution / self.v_resolution

        self.far_plane = far_plane
        self.near_plane = near_plane
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

        self.u = np.zeros(4)  # u is right in camera coords

        self.v = np.zeros(4)  # v is "up" in camera coords

        self.n = np.zeros(4)  # n points at the target from the position

        # for converting world coords to camera space
        self.viewing_matrix = np.ndarray((4, 4))  # Mv

        self.calculate_view_values()

        # for converting camera space to perspective projected + pseudo depth
        self.perspective_matrix = np.ndarray((4, 4))  # Mp

        self.calculate_perspective_matrix()

        # for converting pseudo depth points into screen space points
        self.frustrum_to_screen_transform = np.ndarray((4, 4))

        self.calculate_screen_alignment_matrices()

    def calculate_view_values(self):
        """
        Calculate camera view vectors and world to camera space viewing matrix. Must be recalculated when the camera moves
        """
        n = (self.gaze - self.pos)
        self.n = (n / np.linalg.norm(n))

        u = np.cross(n[:3], Camera.UP[:3])
        self.u = np.hstack((u / np.linalg.norm(u), 0))

        v = np.cross(u[:3], n[:3])
        self.v = np.hstack((v / np.linalg.norm(v), 0))

        self.viewing_matrix = np.linalg.inv(np.hstack(
            (self.n.reshape((4, 1)), self.v.reshape((4, 1)), self.u.reshape((4, 1)), self.pos.reshape((4, 1)))))


    def render_shapes(self, world_shapes: List[generic.GenericShape], light: light_source.LightSource) -> List[RenderedPolygon]:
        faces_out = []
        for i, shape in enumerate(world_shapes):
            for j, face in enumerate(shape.get_world_faces()):
                face_out = face.copy()
                centroid = np.mean(face_out, 0)
                for k, point in enumerate(face):
                    screen_point = np.dot(self.perspective_matrix, np.dot(self.viewing_matrix, point))
                    screen_point /= screen_point[3]
                    screen_point = np.dot(Camera.SWAP_YZ, np.dot(self.frustrum_to_screen_transform, screen_point))
                    face_out[k] = screen_point

                normal = np.hstack((np.cross((face_out[1] - face_out[3])[:3], (face_out[0] - face_out[2])[:3]), 0))

                light_vector = light.pos - centroid
                angle_deviation = max(0, np.dot(normal, light_vector) / (np.linalg.norm(normal) * np.linalg.norm(light_vector)))

                faces_out.append(RenderedPolygon(face_out, (shape.colour[0] * angle_deviation, shape.colour[1] * angle_deviation, shape.colour[2] * angle_deviation)))

        return faces_out

    def calculate_perspective_matrix(self):
        """
        Calculates a matrix that takes points in camera space and perspective projects them onto the near plane with
        pseudo-depth. Remember to divide the operated on point by the depth column afterward
        """

        a = -(self.far_plane + self.near_plane) / (self.far_plane - self.near_plane)

        b = -2 * (self.far_plane * self.near_plane) / (self.far_plane - self.near_plane)

        self.perspective_matrix = np.array(
            (
                (a, 0, 0, b),
                (0, self.near_plane, 0, 0),
                (0, 0, self.near_plane, 0),
                (-1, 0, 0, 0)
            )
        )

    def calculate_screen_alignment_matrices(self):
        """
            Calculates the matrices invovled in projecting a point in the viewing frustrum onto the near plane
        """

        frustrum_top = self.near_plane * np.tan(np.radians(self.fov / 2))

        frustrum_bottom = -frustrum_top

        frustrum_right = self.aspect_ratio * frustrum_top

        frustrum_left = -frustrum_right

        # scales points within the viewing frustrum to the -1 ... 1 scale if they occur within it (further out may be
        # larger and won't show onscreen)
        scale_to_frustrum = np.array(
            (
                (1, 0, 0, 0),
                (0, 2 / (frustrum_top - frustrum_bottom), 0, 0),
                (0, 0, 2 / (frustrum_right - frustrum_left), 0),
                (0, 0, 0, 1)
            )
        )  # S1

        # shifts points from -1 ... 1 scale to the 0 ... 2 scale
        shift_from_negpos_to_0to2 = np.array(
            (
                (1, 0, 0, 0),
                (0, 1, 0, 1),
                (0, 0, 1, 1),
                (0, 0, 0, 1)
            )
        )  # T2

        # scales points from 0 ... 2 to 0 ... width and 0 ... height

        scale_from_0to2_to_widthheight = np.array(
            (
                (1, 0, 0, 0),
                (0, self.v_resolution / 2, 0, 0),
                (0, 0, self.h_resolution / 2, 0),
                (0, 0, 0, 1)
            )
        )  # S2

        self.frustrum_to_screen_transform = np.dot(scale_from_0to2_to_widthheight,
                                                   np.dot(shift_from_negpos_to_0to2, scale_to_frustrum))

    def rotate_left_right(self, amount):
        pass
