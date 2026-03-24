import json
import numpy as np
import robotic as ry


def load_truss_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def get_node_positions(data):

    nodes = {}

    for n in data["node_list"]:
        nid = n["node_id"]
        nodes[nid] = (n["point"]["X"], n["point"]["Y"], n["point"]["Z"])

    return nodes

def get_rods(data):
    elements = {}
    for e in data["element_list"]:
        elements[e["element_id"]] = tuple(e["end_node_ids"])

    return elements

def quaternion_from_z_to_vector(direction):
    direction = np.array(direction)
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
            # same direction → no rotation
            return np.array([1.0, 0.0, 0.0, 0.0])
        else:
            # opposite direction → 180° rotation around x-axis
            return np.array([0.0, 1.0, 0.0, 0.0])

    axis = axis / axis_norm
    angle = np.arccos(np.clip(dot, -1.0, 1.0))

    w = np.cos(angle / 2.0)
    xyz = axis * np.sin(angle / 2.0)

    return np.concatenate(([w], xyz))


def build_truss_in_rai(json_path):

    C = ry.Config()
    C.addFrame("world")
    
    radius = 0.0015  
    data = load_truss_json(json_path)
    node_positions = get_node_positions(data)

    rods = get_rods(data)
    
    for rod_id, (n1, n2) in rods.items():
        p1 = node_positions[n1]
        p2 = node_positions[n2]

        p1 = np.array(node_positions[n1]) * 0.001
        p2 = np.array(node_positions[n2]) * 0.001
    
        length = np.linalg.norm(p2 - p1)
        center = 0.5 * (p1 + p2)
        quat = quaternion_from_z_to_vector(p2 - p1)

        C.addFrame(f"rod_{rod_id}", 'world') .setShape(ry.ST.cylinder, [length, radius]) .setColor([.5,1.,.0]) .setRelativePosition(center) .setQuaternion(quat)
    
    C.view()
    input("Press Enter to close...")

    return C

# creates the rod in rai in the starting position
def create_rod(rod_id, p1, p2):
    radius = 0.0015


build_truss_in_rai('JSON/long_beam_test.json')
