import cv2
import numpy as np
import roma
import torch


class Camera:
    def __init__(self):
        self._rotation = None
        self._translation = None
        self._K = None
        self._parameterization = None

    def _encode_parameterization(self, R, t, rotation_parameterization="rotvec"):
        # convert rotation matrix to correct parameterization
        if rotation_parameterization == "euler":
            rotation = roma.rotmat_to_euler("xyz", R)
        elif rotation_parameterization == "quat":
            rotation = roma.rotmat_to_unitquat(R)
        elif rotation_parameterization == "rotvec":
            rotation = roma.rotmat_to_rotvec(R)
        else:
            raise Exception(f"invalid rotation parameterization {rotation_parameterization}")

        self._parameterization = rotation_parameterization
        self._rotation = rotation
        self._translation = t

    @staticmethod
    def from_Rt(K, R_t, parameterization="rotvec"):
        """:K: camera intrinsics :R_t: world2cam projection matrix (either 3x4 or 4x4)
        :parameterization: the way the camera is supposed to be represented."""
        camera = Camera()
        R_t = torch.from_numpy(R_t) if isinstance(R_t, np.ndarray) else R_t
        camera._encode_parameterization(
            R=R_t[:3, :3], t=R_t[:3, 3], rotation_parameterization=parameterization
        )
        camera.K = torch.from_numpy(K) if isinstance(K, np.ndarray) else K
        return camera

    @staticmethod
    def from_params(rotation, translation, K, parameterization="rotvec"):
        """:rotation: the rotation parameters in the form of parameterization :translation: the
        translation :K: the K :parameterization: the parameteization of the rotation parameters."""
        camera = Camera()
        camera._rotation = rotation
        camera._translation = translation.squeeze()
        camera._K = K
        camera._parameterization = parameterization
        return camera

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, rotation):
        self._rotation = rotation

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, translation):
        self._translation = translation

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, K):
        self._K = K

    @property
    def camera_center(self):
        return self.R.T @ -self.t

    @property
    def R(self):
        if self._parameterization == "euler":
            R = roma.euler_to_rotmat("xyz", self._rotation)
        elif self._parameterization == "quat":
            R = roma.unitquat_to_rotmat(self._rotation)
        elif self._parameterization == "rotvec":
            R = roma.rotvec_to_rotmat(self._rotation)
        return R

    @property
    def P(self):
        return torch.hstack((self.R, self.t))

    @property
    def t(self):
        return self.translation.reshape(3, 1)

    @property
    def parameterization(self):
        return self._parameterization

    @property
    def K_inv(self):
        return np.linalg.inv(self._K)

    @property
    def R_t(self):
        return self._R_t

    @R_t.setter
    def R_t(self, R_t):
        self._R_t = R_t

    @property
    def R_t_inv(self):
        # calculate inverse
        R_t_inv = np.eye(4)
        R_t_inv[:3, :3] = self._R_t[:3, :3].T
        R_t_inv[:3, 3] = R_t_inv[:3, :3] @ -self._R_t[:3, 3]
        return R_t_inv

    # @property
    # def P(self):
    #     return Camera.compose_P(R_t=self._R_t, K=self._K)

    # static methods
    @staticmethod
    def compose_K(f_x, f_y, c_x, c_y):
        if isinstance(f_x, np.ndarray):
            return np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
        else:
            return torch.tensor([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])

    @staticmethod
    def compose_P(R=None, t=None, R_t=None, K=None, homo_out=True):
        if isinstance(R, np.ndarray) or isinstance(R_t, np.ndarray):
            R_t_homo = np.eye(4)
            if R_t is None:
                R_t_homo[:3, :3] = R[:3, :3]
                R_t_homo[:3, 3] = t[:3]
            else:
                R_t_homo = np.eye(4)
                R_t_homo[:3, :] = R_t[:3, :]

            # convert in case homogeneous coordinates are used
            K_homo = np.eye(4)
            K_homo[:3, :3] = K
            if homo_out:
                return np.dot(K_homo, R_t_homo)
            else:
                return np.dot(K_homo, R_t_homo)[:3, :]
        else:
            R_t_homo = torch.eye(4)
            if R_t is None:
                R_t_homo[:3, :3] = R[:3, :3]
                R_t_homo[:3, 3] = t[:3]
            else:
                R_t_homo = torch.eye(4)
                R_t_homo[:3, :] = R_t[:3, :]

            # convert in case homogeneous coordinates are used
            K_homo = torch.eye(4)
            K_homo[:3, :3] = K
            if homo_out:
                return K_homo @ R_t_homo
            else:
                return (K_homo @ R_t_homo)[:3, :]

    # look at method
    @staticmethod
    def look_at_matrix(eye, center, up):
        """Creates a world to camera matrix given the eye, center and up vectors.

        Args:
            eye (torch.Tensor): position of the camera
            center (torch.Tensor): position to "look at"
            up (torch.Tensor): the up vector

        Returns:
            torch.Tensor: world to camera matrix 4x4
        """
        forward = center - eye  # since
        forward = forward / torch.norm(forward)
        side = torch.linalg.cross(forward, up)
        side = side / torch.norm(side)
        up = torch.linalg.cross(side, forward)
        up = up / torch.norm(up)

        m = torch.eye(4)
        m[:3, :3] = torch.stack([side, up, -forward]).T  # w2c
        m[:3, 3] = m[:3, :3] @ (-eye)  # t = R @ (-eye)
        return m

    @staticmethod
    def K_to_pixel_dimensions(K, m_x, m_y):
        K = K.clone() if isinstance(K, torch.Tensor) else K.copy()
        K[0, 0] = K[0, 0] * m_x
        K[0, 2] = K[0, 2] * m_x
        K[1, 1] = K[1, 1] * m_y
        K[1, 2] = K[1, 2] * m_y
        return K

    @staticmethod
    def K_to_unit_dimensions(K, m_x, m_y):
        K = K.clone()
        K[0, 0] = K[0, 0] / m_x
        K[0, 2] = K[0, 2] / m_x
        K[1, 1] = K[1, 1] / m_y
        K[1, 2] = K[1, 2] / m_y
        return K

    @staticmethod
    def to_R(x, rotation_parameterization="rotvec"):
        if rotation_parameterization == "euler":
            return roma.euler_to_rotmat("xyz", x)
        elif rotation_parameterization == "quat":
            return roma.unitquat_to_rotmat(x)
        elif rotation_parameterization == "rotvec":
            return roma.rotvec_to_rotmat(x)
        elif rotation_parameterization == "rotmat":
            return x
        else:
            raise ValueError(f"Unknown camera parameterization: {rotation_parameterization}")

    @staticmethod
    def geodesic_distance(
        R1, R2, rotation_parameterization_1="euler", rotation_parameterization_2="euler"
    ):
        R1 = Camera.to_R(R1, rotation_parameterization_1)
        R2 = Camera.to_R(R2, rotation_parameterization_2)
        return roma.rotmat_geodesic_distance(R1, R2)

    @staticmethod
    def euler_to_R(theta):
        """Converts euler angles to a rotation matrix.

        Args:
            theta (torch.Tensor): euler angles in radians

        Returns:
            torch.Tensor: rotation matrix
        """
        R_x = torch.stack(
            [
                torch.tensor([1, 0, 0]),
                torch.tensor([0, torch.cos(theta[0]), -torch.sin(theta[0])]),
                torch.tensor([0, torch.sin(theta[0]), torch.cos(theta[0])]),
            ]
        )
        R_y = torch.stack(
            [
                torch.tensor([torch.cos(theta[1]), 0, torch.sin(theta[1])]),
                torch.tensor([0, 1, 0]),
                torch.tensor([-torch.sin(theta[1]), 0, torch.cos(theta[1])]),
            ]
        )
        R_z = torch.stack(
            [
                torch.tensor([torch.cos(theta[2]), -torch.sin(theta[2]), 0]),
                torch.tensor([torch.sin(theta[2]), torch.cos(theta[2]), 0]),
                torch.tensor([0, 0, 1]),
            ]
        )
        return R_z @ R_y @ R_x

    @staticmethod
    def R_to_euler(R):
        """Converts a rotation matrix to euler angles.

        Args:
            R (torch.Tensor): rotation matrix

        Returns:
            torch.Tensor: euler angles in radians
        """
        sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        x = torch.atan2(R[2, 1], R[2, 2])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.atan2(R[1, 0], R[0, 0])
        if singular:
            print("ALERT: Singular")
            # print(R)
            x = torch.atan2(-R[1, 2], R[1, 1])
            y = torch.atan2(-R[2, 0], sy)
            z = 0
        return torch.tensor([x, y, z])

    ####
    # Quaternion Stuff
    ####

    # invert quaternion
    def _quat_inv(q):
        # assumes real part is first and quaternion is unit
        return torch.cat(
            [
                q[0].unsqueeze(0),
                q[1:] * -1,
            ],
            dim=0,
        )

    def _quat_mult(q1, q2):
        t1 = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
        t2 = q1[0] * q2[1] + q1[1] * q2[0] - q1[2] * q2[3] + q1[3] * q2[2]
        t3 = q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0] - q1[3] * q2[1]
        t4 = q1[0] * q2[3] - q1[1] * q2[2] + q1[2] * q2[1] + q1[3] * q2[0]
        return torch.tensor([t1, t2, t3, t4])

    @staticmethod
    def quat_to_R_1(q):
        a, b, c, d = q
        return torch.stack(
            [
                torch.stack(
                    [
                        a**2 + b**2 - c**2 - d**2,
                        2 * b * c - 2 * a * d,
                        2 * b * d + 2 * a * c,
                    ]
                ),
                torch.stack(
                    [
                        2 * b * c + 2 * a * d,
                        a**2 - b**2 + c**2 - d**2,
                        2 * c * d - 2 * a * b,
                    ]
                ),
                torch.stack(
                    [
                        2 * b * d - 2 * a * c,
                        2 * c * d + 2 * a * b,
                        a**2 - b**2 - c**2 + d**2,
                    ]
                ),
            ]
        )

    @staticmethod
    def quat_to_R_2(q):
        R = torch.zeros((3, 3))
        R[0, 0] += 1 - 2 * (q[2] ** 2 + q[3] ** 2)
        R[0, 1] += 2 * (q[1] * q[2] - q[0] * q[3])
        R[0, 2] += 2 * (q[0] * q[2] + q[1] * q[3])
        R[1, 0] += 2 * (q[1] * q[2] + q[0] * q[3])
        R[1, 1] += 1 - 2 * (q[1] ** 2 + q[3] ** 2)
        R[1, 2] += 2 * (q[2] * q[3] - q[0] * q[1])
        R[2, 0] += 2 * (q[1] * q[3] - q[0] * q[2])
        R[2, 1] += 2 * (q[0] * q[1] + q[2] * q[3])
        R[2, 2] += 1 - 2 * (q[1] ** 2 + q[2] ** 2)
        return R

    @staticmethod
    def quat_rotate_3(q, p):
        return Camera._quat_mult(Camera._quat_mult(Camera._quat_inv(q), p), q)[1:]

    @staticmethod
    def R_to_quat(R):
        # find magnitude of each quaternion component
        mag_q0 = torch.sqrt((1 + R[0, 0] + R[1, 1] + R[2, 2]) / 4)
        mag_q1 = torch.sqrt((1 + R[0, 0] - R[1, 1] - R[2, 2]) / 4)
        mag_q2 = torch.sqrt((1 - R[0, 0] + R[1, 1] - R[2, 2]) / 4)
        mag_q3 = torch.sqrt((1 - R[0, 0] - R[1, 1] + R[2, 2]) / 4)

        # find largest magnitude
        if mag_q0 >= mag_q1 and mag_q0 >= mag_q2 and mag_q0 >= mag_q3:
            q0 = mag_q0
            q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
            q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
            q3 = (R[1, 0] - R[0, 1]) / (4 * q0)
        elif mag_q1 >= mag_q0 and mag_q1 >= mag_q2 and mag_q1 >= mag_q3:
            q1 = mag_q1
            q0 = (R[2, 1] - R[1, 2]) / (4 * q1)
            q2 = (R[1, 0] + R[0, 1]) / (4 * q1)
            q3 = (R[0, 2] + R[2, 0]) / (4 * q1)
        elif mag_q2 >= mag_q0 and mag_q2 >= mag_q1 and mag_q2 >= mag_q3:
            q2 = mag_q2
            q0 = (R[0, 2] - R[2, 0]) / (4 * q2)
            q1 = (R[1, 0] + R[0, 1]) / (4 * q2)
            q3 = (R[2, 1] + R[1, 2]) / (4 * q2)
        else:
            q3 = mag_q3
            q0 = (R[1, 0] - R[0, 1]) / (4 * q3)
            q1 = (R[0, 2] + R[2, 0]) / (4 * q3)
            q2 = (R[2, 1] + R[1, 2]) / (4 * q3)

        return torch.tensor([q0, q1, q2, q3])


def nerf_to_colmap_extrinsics(R_nerf, t_nerf):
    """Returns the *c2w* coordinates in COLMAP/OpenCV camera model."""
    # Define the adjustment matrix
    M_adjust = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

    M_adjust2 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # Convert rotation matrix
    R_colmap = (M_adjust @ R_nerf) @ M_adjust2

    # Convert translation vector
    t_colmap = M_adjust @ t_nerf

    return R_colmap, t_colmap


def undistort_fisheye_intrinsics(data, center_principal_point=True):
    """see also: https://github.com/scannetpp/scannetpp/blob/main/dslr/undistort.py"""
    K = data.K
    distortion = data.params[4:]
    dim = (data.width, data.height)

    K_undistorted = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, distortion, dim, np.eye(3), balance=0
    )

    if center_principal_point:  # Make the cx and cy to be the center of the image
        K_undistorted[0, 2] = dim[0] / 2.0
        K_undistorted[1, 2] = dim[1] / 2.0
        return K_undistorted
    return K_undistorted
