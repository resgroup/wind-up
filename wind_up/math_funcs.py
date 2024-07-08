import numpy as np
import numpy.typing as npt


def circ_diff(angle1: float | npt.NDArray | list, angle2: float | npt.NDArray | list) -> float | npt.NDArray:
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

    return np.mod(angle1 - angle2 + 180, 360) - 180
