"""
As in https://github.com/MichaelGrupp/evo
See https://github.com/MichaelGrupp/evo/blob/051e5bf63195172af58dc8256cc71618f079f224/notebooks/metrics.py_API_Documentation.ipynb
for the formulas.
"""

import numpy as np

from ATE.trajectory_utils import compute_angle


def relative_transformation(P_ref, P_est):
    return np.linalg.inv(P_ref) @ P_est


class AbsolutePoseError:
    @staticmethod
    def full_transformation_error(P_ref, P_est):
        E = relative_transformation(P_ref, P_est)
        return np.linalg.norm(E - np.eye(4))

    @staticmethod
    def translation_error(t_ref, t_est):
        return np.linalg.norm(t_ref - t_est)

    @staticmethod
    def rotation_error(P_ref, P_est):
        E = relative_transformation(P_ref, P_est)
        return np.linalg.norm(E[:3, :3] - np.eye(3))

    @staticmethod
    def angle_error(P_ref, P_est):
        E = relative_transformation(P_ref, P_est)
        return abs(compute_angle(E))
