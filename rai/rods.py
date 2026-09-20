# handles all rod stuff
# can be replaced in the future for the actual rods

import numpy as np
import robotic as ry
from .utils import quaternion_from_z_to_vector
import time

class RodManager:
    def __init__(self, C, truss, radius=0.0015, scale=0.001):
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

    @staticmethod
    def _closest_points_on_segments(p1, p2, q1, q2):
        """Return the closest points on two finite line segments."""
        rod_direction = p2 - p1
        other_direction = q2 - q1
        offset = p1 - q1

        rod_length_squared = np.dot(rod_direction, rod_direction)
        other_length_squared = np.dot(other_direction, other_direction)

        if rod_length_squared < 1e-12 or other_length_squared < 1e-12:
            raise ValueError("Cannot calculate a coupler direction for a zero-length rod")

        cross_term = np.dot(rod_direction, other_direction)
        rod_offset = np.dot(rod_direction, offset)
        other_offset = np.dot(other_direction, offset)
        denominator = (
            rod_length_squared * other_length_squared
            - cross_term * cross_term
        )

        if denominator > 1e-12:
            rod_fraction = np.clip(
                (
                    cross_term * other_offset
                    - other_length_squared * rod_offset
                ) / denominator,
                0.0,
                1.0,
            )
        else:
            rod_fraction = 0.5

        other_fraction = np.clip(
            (
                cross_term * rod_fraction
                + other_offset
            ) / other_length_squared,
            0.0,
            1.0,
        )
        rod_fraction = np.clip(
            (
                cross_term * other_fraction
                - rod_offset
            ) / rod_length_squared,
            0.0,
            1.0,
        )

        return (
            p1 + rod_fraction * rod_direction,
            q1 + other_fraction * other_direction,
        )

    def get_coupler_pointing_direction(
        self,
        rod_id,
        connected_rod_ids=None,
        grasp_fraction=0.5,
    ):
        """
        Return the gripper pointing direction for placing a rod.

        Grounded rods are approached downward. Other rods are approached from
        the side opposite an already coupled rod. When several active couplers
        exist, use the one nearest the gripper's position along the candidate
        rod. This avoids creating an artificial direction between couplers on
        different sides of the rod.
        """
        p1, p2 = self.get_rod_endpoints(rod_id)
        rod_vector = p2 - p1
        rod_length = np.linalg.norm(rod_vector)
        rod_axis = rod_vector / rod_length

        if rod_id in self.truss.grounded_rods:
            pointing_direction = np.array([0.0, 0.0, -1.0])
        else:
            active_rods = (
                None
                if connected_rod_ids is None
                else set(connected_rod_ids)
            )
            coupled_rods = sorted(
                rod_2 if rod_1 == rod_id else rod_1
                for rod_1, rod_2 in self.truss.couplers
                if (
                    rod_id in (rod_1, rod_2)
                    and (
                        active_rods is None
                        or (
                            rod_2 if rod_1 == rod_id else rod_1
                        ) in active_rods
                    )
                )
            )

            coupler_directions = []

            for coupled_rod_id in coupled_rods:
                q1, q2 = self.get_rod_endpoints(coupled_rod_id)
                rod_point, coupled_point = self._closest_points_on_segments(
                    p1,
                    p2,
                    q1,
                    q2,
                )
                direction = coupled_point - rod_point

                # The rod axis already fixes the gripper's local X-axis. Keep
                # the pointing direction perpendicular to it so the two hard
                # orientation constraints remain compatible.
                direction -= np.dot(direction, rod_axis) * rod_axis
                direction_norm = np.linalg.norm(direction)

                if direction_norm > 1e-8:
                    coupler_fraction = np.clip(
                        np.dot(rod_point - p1, rod_axis) / rod_length,
                        0.0,
                        1.0,
                    )
                    coupler_directions.append((
                        abs(coupler_fraction - grasp_fraction),
                        coupled_rod_id,
                        direction / direction_norm,
                    ))

            if not coupler_directions:
                raise ValueError(
                    f"Rod {rod_id} has no usable active coupler direction."
                )

            coupler_directions.sort(key=lambda item: (item[0], item[1]))
            pointing_direction = coupler_directions[0][2]

        pointing_direction -= (
            np.dot(pointing_direction, rod_axis) * rod_axis
        )
        direction_norm = np.linalg.norm(pointing_direction)

        if direction_norm < 1e-8:
            raise ValueError(
                f"Rod {rod_id} pointing direction is parallel to its axis."
            )

        return pointing_direction / direction_norm

    def create_gripper_direction_target(
        self,
        rod_id,
        connected_rod_ids=None,
        grasp_fraction=0.5,
        target_suffix=None,
    ):
        """Create a world frame whose local Z-axis points toward the coupler."""
        direction = self.get_coupler_pointing_direction(
            rod_id,
            connected_rod_ids=connected_rod_ids,
            grasp_fraction=grasp_fraction,
        )
        p1, p2 = self.get_rod_endpoints(rod_id)
        suffix = f"_{target_suffix}" if target_suffix else ""
        target_name = f"rod_{rod_id}_gripper_direction_target{suffix}"

        if target_name not in self.C.getFrameNames():
            self.C.addFrame(target_name, "world")

        self.C.getFrame(target_name) \
            .setPosition(0.5 * (p1 + p2)) \
            .setQuaternion(quaternion_from_z_to_vector(direction))

        return target_name, direction
    
    
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
    
    def create_rod_at_goal_pose(self, rod_id):
        center, quat = self.get_goal_pose(rod_id)
        
        quat = np.array(quat, dtype=float)
        quat = quat / np.linalg.norm(quat)


        n1, n2 = self.truss.elements[rod_id]

        p1 = np.array(self.truss.nodes[n1], dtype=float) * self.scale
        p2 = np.array(self.truss.nodes[n2], dtype=float) * self.scale

        length = np.linalg.norm(p2 - p1) -0.03 #-0.03 for long_beam
        
        if length < 1e-10:
            raise ValueError(f"Rod {rod_id} has zero length")

        self.C.addFrame(f"rod_{rod_id}") .setShape(ry.ST.cylinder, [length, self.radius]) .setColor([.5,1.,.0]) .setPosition(center) .setQuaternion(quat) .setContact(1)
    
        return
    
    def create_target_frame(self, rod_id):
        center, quat = self.get_goal_pose(rod_id)

        target_name = f"rod_{rod_id}_target"
        if target_name not in self.C.getFrameNames():
            self.C.addFrame(target_name, "world")

        self.C.getFrame(target_name).setPosition(center).setQuaternion(quat)
        return target_name
        
    def create_dual_arm_grasp_frames(
        self,
        rod_id,
        d1_from_end=0.04,
        d12_between_arms=0.12,
        frame_suffix=None,
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

        suffix = f"_{frame_suffix}" if frame_suffix else ""
        g1 = f"rod_{rod_id}_grasp_a1{suffix}"
        g2 = f"rod_{rod_id}_grasp_a2{suffix}"

        if g1 not in self.C.getFrameNames():
            self.C.addFrame(g1, rod)

        if g2 not in self.C.getFrameNames():
            self.C.addFrame(g2, rod)

        self.C.getFrame(g1).setRelativePosition([0.0, 0.0, z1])
        self.C.getFrame(g2).setRelativePosition([0.0, 0.0, z2])

        return g1, g2
    
    def set_to_goal_pose(self, rod_id, view=False):
        center, quat = self.get_goal_pose(rod_id)

        self.C.getFrame(f"rod_{rod_id}") \
            .setPosition(center) \
            .setQuaternion(quat)

        if view:
            self.C.view()
            time.sleep(2)
               

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

        if frame_name not in self.C.getFrameNames():
            self.C.addFrame(frame_name, rod)

        self.C.getFrame(frame_name).setRelativePosition([0.0, 0.0, z])

        return frame_name
