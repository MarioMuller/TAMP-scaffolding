# utility functions are placed here

import numpy as np

def quaternion_from_z_to_vector(direction):
    direction = np.asarray(direction)
    direction = direction / np.linalg.norm(direction)

    z = np.array([0.0, 0.0, 1.0])

    # cross product gives rotation axis
    axis = np.cross(z, direction)
    axis_norm = np.linalg.norm(axis)

    # dot product gives cos(angle)
    dot = np.dot(z, direction)

    # handle parallel case
    if axis_norm < 1e-8:
        if dot > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        else:
            return np.array([0.0, 1.0, 0.0, 0.0])

    axis = axis / axis_norm
    angle = np.arccos(np.clip(dot, -1.0, 1.0))

    w = np.cos(angle / 2.0)
    xyz = axis * np.sin(angle / 2.0)

    return np.concatenate(([w], xyz))