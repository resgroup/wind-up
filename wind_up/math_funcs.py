import numpy as np


def circ_diff(angle1: float | np.generic, angle2: float | np.generic) -> float | np.generic:
    angle1_rad = np.radians(angle1)
    angle2_rad = np.radians(angle2)
    sin_angle1 = np.sin(angle1_rad)
    cos_angle1 = np.cos(angle1_rad)
    sin_angle2 = np.sin(angle2_rad)
    cos_angle2 = np.cos(angle2_rad)
    denominator = cos_angle2 * cos_angle2 + sin_angle2 * sin_angle2
    temp1 = (cos_angle1 * cos_angle2 + sin_angle1 * sin_angle2) / denominator
    temp2 = (sin_angle1 * cos_angle2 - cos_angle1 * sin_angle2) / denominator
    return np.degrees(np.arctan2(temp2, temp1))
