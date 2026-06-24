# based on https://github.com/yijiangh/husky_assembly_tamp/blob/yh/dual_arm_integrate/husky_assembly_tamp/utils/util.py

from typing import List, Union, Tuple
import numpy as np


def closest_points_between_segments(seg1: List[Union[List[float], np.ndarray]], seg2: List[Union[List[float], np.ndarray]]) -> Tuple[List[float], List[float]]:
    """
    Calculate the endpoints of the common perpendicular line between two line segments.

    Params:
        seg1 ([[x1, y1, z1], [x2, y2, z2]]): segment 1
        seg2 ([[x1, y1, z1], [x2, y2, z2]]): segment 2

    Returns:
        [x1, y1, z1]: point on segment 1
        [x2, y2, z2]: point on segment 1

    """
    p1, q1 = np.array(seg1[0]), np.array(seg1[1])
    p2, q2 = np.array(seg2[0]), np.array(seg2[1])

    d1 = q1 - p1
    d2 = q2 - p2

    r = p1 - p2

    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, r)
    e = np.dot(d2, r)

    denom = a * c - b * b
    if denom == 0:
        raise ValueError("Segments are parallel!")

    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom

    s = np.clip(s, 0, 1)
    t = np.clip(t, 0, 1)

    closest_point_on_seg1: np.ndarray = p1 + s * d1
    closest_point_on_seg2: np.ndarray = p2 + t * d2

    return closest_point_on_seg1.tolist(), closest_point_on_seg2.tolist()