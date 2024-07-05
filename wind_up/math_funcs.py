import numpy as np


def circ_diff(angle1: float | np.generic | list, angle2: float | np.generic | list) -> float | np.generic:
    """Calculate the circular difference between two angles.

    Args:
        angle1: First angle in degrees.
        angle2: Second angle in degrees.

    Returns:
        Circular difference between the two angles in degrees
    """

    # Convert list to numpy array
    if isinstance(angle1, list):
        angle1 = np.array(angle1)
    if isinstance(angle2, list):
        angle2 = np.array(angle2)

    return np.mod(angle1 - angle2 + 180.0, 360.0) - 180.0
