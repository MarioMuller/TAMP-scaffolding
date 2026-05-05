# handles all rod stuff
# can be replaced in the future for the actual rods

import numpy as np
import robotic as ry
from .utils import quaternion_from_z_to_vector
import time

class RodManager:
    def __init__(self, C, truss, radius=0.0015, scale=0.00351):
        self.C = C
        self.truss = truss
        self.radius = radius
        self.scale = scale
        
    def get_rod_endpoints(self, rod_id):
        n1, n2 = self.truss.elements[rod_id]
        p1 = np.asarray(self.truss.nodes[n1], dtype=float) * self.scale
        p2 = np.asarray(self.truss.nodes[n2], dtype=float) * self.scale
        return p1, p2    
        
    def get_goal_pose(self, rod_id):
        
        p1, p2 = self.get_rod_endpoints(rod_id)
        center = 0.5 * (p1 + p2)
        center = center + [0, 0, 0.1]
        quat = quaternion_from_z_to_vector(p2 - p1)

        return center, quat    
        
    def get_rod_length(self, rod_id):
        
        p1, p2 = self.get_rod_endpoints(rod_id)

        # same shortening as in create_rod()
        return np.linalg.norm(p2 - p1) - 0.03    
    
    
    # creates the next required rod
    def create_rod(self, rod_id, pos = [-0.4,-0.05,0.2], ori = [0.7070, 1, 0, 0.7070]):
        
        ori = np.array(ori, dtype=float)
        ori = ori / np.linalg.norm(ori)


        n1, n2 = self.truss.elements[rod_id]

        p1 = np.array(self.truss.nodes[n1], dtype=float) * self.scale
        p2 = np.array(self.truss.nodes[n2], dtype=float) * self.scale

        length = np.linalg.norm(p2 - p1) -0.03 #-0.03 for long_beam
        
        if length < 1e-10:
            raise ValueError(f"Rod {rod_id} has zero length")

        self.C.addFrame(f"rod_{rod_id}") .setShape(ry.ST.cylinder, [length, self.radius]) .setColor([.5,1.,.0]) .setPosition(pos) .setQuaternion(ori) .setContact(1)
    
        return
    
    def create_target_frame(self, rod_id):
        center, quat = self.get_goal_pose(rod_id)

        target_name = f"rod_{rod_id}_target"
        if self.C.getFrame(target_name) is None:
            self.C.addFrame(target_name, "world")

        self.C.getFrame(target_name).setPosition(center).setQuaternion(quat)
        return target_name
        
    def create_dual_arm_grasp_frames(
        self,
        rod_id,
        d1_from_end=0.04,
        d12_between_arms=0.12,
    ):
        """
        Creates two grasp frames fixed on the rod.

        Assumption:
        - RAI cylinder local z-axis is the rod axis.
        - d1_from_end is measured from rod negative-z end.
        - d12_between_arms is measured along the rod axis.
        """

        rod = f"rod_{rod_id}"
        length = self.get_rod_length(rod_id)

        d2_from_end = d1_from_end + d12_between_arms

        if d1_from_end < 0.0 or d2_from_end > length:
            raise ValueError(
                f"Invalid grasp distances: d1={d1_from_end}, d2={d2_from_end}, rod length={length}"
            )

        z1 = -0.5 * length + d1_from_end
        z2 = -0.5 * length + d2_from_end

        g1 = f"rod_{rod_id}_grasp_a1"
        g2 = f"rod_{rod_id}_grasp_a2"

        if self.C.getFrame(g1) is None:
            self.C.addFrame(g1, rod)

        if self.C.getFrame(g2) is None:
            self.C.addFrame(g2, rod)

        self.C.getFrame(g1).setRelativePosition([0.0, 0.0, z1])
        self.C.getFrame(g2).setRelativePosition([0.0, 0.0, z2])

        return g1, g2
    
    # helper function to move rods out of the way 
    def set_to_end_position(self, rod_id):
        
        n1, n2 = self.truss.elements[rod_id]

        p1 = np.array(self.truss.nodes[n1]) * self.scale
        p2 = np.array(self.truss.nodes[n2]) * self.scale
    
        center = 0.5 * (p1 + p2)
        center[2] += 0.1
        quat = quaternion_from_z_to_vector(p2 - p1)

        self.C.getFrame(f"rod_{rod_id}").setPosition(center) .setQuaternion(quat)
        
        self.C.view()
        # input("Press Enter to close...")

        return
    
    def set_to_goal_pose(self, rod_id, view=False):
        center, quat = self.get_goal_pose(rod_id)

        self.C.getFrame(f"rod_{rod_id}") \
            .setPosition(center) \
            .setQuaternion(quat)

        if view:
            self.C.view()
            
            
    def create_sliding_support_grasp_frame(self, rod_id):
        rod = f"rod_{rod_id}"
        length = self.get_rod_length(rod_id)

        frame_name = f"rod_{rod_id}_support_grasp"

        if self.C.getFrame(frame_name) is None:
            self.C.addFrame(frame_name, rod) \
                .setJoint(
                    ry.JT.transZ,
                    limits=np.array([-0.5 * length, 0.5 * length])
                )

        return frame_name       

    def create_support_grasp_frame_at_fraction(self, rod_id, fraction):
        """
        Creates a fixed support grasp frame at a fraction along the rod.

        fraction:
            0.0 = one end
            0.5 = middle
            1.0 = other end
        """

        if fraction < 0.0 or fraction > 1.0:
            raise ValueError("fraction must be between 0.0 and 1.0")

        rod = f"rod_{rod_id}"
        length = self.get_rod_length(rod_id)

        z = -0.5 * length + fraction * length

        frame_name = f"rod_{rod_id}_support_grasp_{fraction:.2f}"

        if self.C.getFrame(frame_name) is None:
            self.C.addFrame(frame_name, rod)

        self.C.getFrame(frame_name).setRelativePosition([0.0, 0.0, z])

        return frame_name
        