from json_import import Truss
from Simple_search import TrussSearch
import numpy as np
import robotic as ry

class RaiTrussBuilder:

    def __init__(self, truss, radius=0.0015):
        self.truss = truss
        self.radius = radius
        self.C = ry.Config()
        self.C.addFrame("world")

    def quaternion_from_z_to_vector(self, direction):
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


    def build_entire_truss_in_rai(self):

        
        for rod_id, (n1, n2) in self.truss.elements.items():

            p1 = np.array(self.truss.nodes[n1]) * 0.001
            p2 = np.array(self.truss.nodes[n2]) * 0.001
        
            length = np.linalg.norm(p2 - p1)
            center = 0.5 * (p1 + p2)
            quat = self.quaternion_from_z_to_vector(p2 - p1)

            C.addFrame(f"rod_{rod_id}", 'world') .setShape(ry.ST.cylinder, [length, self.radius]) .setColor([.5,1.,.0]) .setRelativePosition(center) .setQuaternion(quat)
        
        C.view()
        input("Press Enter to close...")

        return C

    # creates the next required rod
    def create_rod(self, rod_id):

        n1, n2 = self.truss.elements[rod_id]

        p1 = np.array(self.truss.nodes[n1], dtype=float) * 0.001
        p2 = np.array(self.truss.nodes[n2], dtype=float) * 0.001

        length = np.linalg.norm(p2 - p1)
        if length < 1e-10:
            raise ValueError(f"Rod {rod_id} has zero length")

        C.addFrame(f"rod_{rod_id}", 'world') .setShape(ry.ST.cylinder, [length, self.radius]) .setColor([.5,1.,.0]) .setRelativePosition([0,1,0])
        C.view()
        input("Press Enter to close...")
        return C
    

if __name__ == "__main__":
    C = ry.Config()
    C.addFrame("world")

    truss = Truss.from_json("JSON/long_beam_test.json")

    # build_entire_truss_in_rai(radius, node_positions, rods, C)
    builder = RaiTrussBuilder(truss, radius=0.0015)
    builder.create_rod(1)
    builder.build_entire_truss_in_rai()