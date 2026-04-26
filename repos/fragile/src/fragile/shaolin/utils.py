import tempfile

import numpy as np
from PIL import Image


def find_closest_point(points: np.ndarray, x: float, y: float) -> int:
    """Find the index of the point closest to the target.

    This function calculates the Euclidean distance from the target to each point
    in the array and returns the index of the point with the minimum distance.

    Args:
        points (np.ndarray): An array of points.
        x (float): The x coordinate of the target.
        y (float): The y coordinate of the target.

    Returns:
        int: The index of the point closest to the target.

    """
    # Calculate the Euclidean distance from the target to each point
    target = np.array([[x, y]])
    distances = np.sqrt(np.sum((points - target) ** 2, axis=1))
    # Find the index of the minimum distance
    return np.argmin(distances)


def create_gif(data, filename=None, fps=10, optimize=False):
    """Create a GIF from a list of images."""
    duration = int((len(data) / fps) * 20)
    filename = tempfile.NamedTemporaryFile(suffix="a.gif") if filename is None else filename
    images = [Image.fromarray(v) for v in data]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        optimize=optimize,
        duration=duration,
        loop=0,
    )
    return filename
