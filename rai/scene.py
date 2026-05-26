import os
import numpy as np
import robotic as ry
import time


class RaiScene:
    def __init__(self):
        self.C = ry.Config()
        self.C.addFrame("world")

    def clear(self):
        self.C.clear()
        self.C.addFrame("world")

    def _has_frame(self, name):
        if hasattr(self.C, "getFrameNames"):
            return name in self.C.getFrameNames()

        return self.C.getFrame(name) is not None

    def _ensure_table(self):
        if self._has_frame("table"):
            return

        self.C.addFrame("table") \
            .setPosition([0, 0, 0.0]) \
            .setShape(ry.ST.box, size=[20, 20, 0.02, 0.005]) \
            .setColor([0.9, 0.9, 0.9]) \
            .setContact(1)

    def import_husky(self):

        self._ensure_table()

        # paths to the files
        husky_path = os.path.join(os.path.dirname(__file__), "../src/models/husky/husky.g")
        robot_path = os.path.join(os.path.dirname(__file__), "../src/models/ur5/ur5.g")


        self.C.addFrame("husky_base_XYPhi_joint") .setParent(self.C.getFrame("world")) .setJoint(
            ry.JT.transXYPhi, limits=np.array([-30, 30, -30, 30, -3.14, 3.14])
        ).setJointState([-1., 0, 0])

        self.C.addFile(husky_path, namePrefix="husky_coll_").setParent(
            self.C.getFrame("husky_base_XYPhi_joint")
        ).setRelativePosition([0, 0.0, 0.16])

        # attatch both arms to the husky
        self.C.addFile(robot_path, namePrefix="a1_").setParent(
        self.C.getFrame("husky_coll_right_arm_bulkhead_joint")
            ).setRelativePosition([0, 0, 0]).setRelativeQuaternion([1, 0, 0, 0])

        self.C.addFile(robot_path, namePrefix="a2_").setParent(
        self.C.getFrame("husky_coll_left_arm_bulkhead_joint")
            ).setRelativePosition([0, 0, 0]).setRelativeQuaternion([1, 0, 0, 0])

        return

    def import_support_husky(
        self,
        name="h2",
        base_q=(3.0, -3.0, 0.0),
        arm="right",  # or "left"
    ):
        """
        Import a Husky with a single arm.

        name:
            prefix to uniquely identify robot

        """

        self._ensure_table()

        husky_path = os.path.join(
            os.path.dirname(__file__),
            "../src/models/husky/husky.g",
        )

        robot_path = os.path.join(
            os.path.dirname(__file__),
            "../src/models/ur5/ur5.g",
        )

        base_frame = f"{name}_base_XYPhi_joint"
        husky_prefix = f"{name}_husky_"

        # base
        self.C.addFrame(base_frame) \
            .setParent(self.C.getFrame("world")) \
            .setJoint(
                ry.JT.transXYPhi,
                limits=np.array([-30, 30, -30, 30, -3.14, 3.14]),
            ) \
            .setJointState(list(base_q))

        # husky body
        self.C.addFile(husky_path, namePrefix=husky_prefix) \
            .setParent(self.C.getFrame(base_frame)) \
            .setRelativePosition([0, 0.0, 0.16])

        # choose which arm
        if arm == "right":
            parent_joint = f"{husky_prefix}right_arm_bulkhead_joint"
            arm_prefix = f"{name}_a1_"

        elif arm == "left":
            parent_joint = f"{husky_prefix}left_arm_bulkhead_joint"
            arm_prefix = f"{name}_a1_"

        else:
            raise ValueError("arm must be 'right' or 'left'")

        # attach single arm
        self.C.addFile(robot_path, namePrefix=arm_prefix) \
            .setParent(self.C.getFrame(parent_joint)) \
            .setRelativePosition([0, 0, 0]) \
            .setRelativeQuaternion([1, 0, 0, 0])


    def import_floating_grippers_debug(self):

        self._ensure_table()

        robotiq_path = os.path.join(
            os.path.dirname(__file__),
            "../src/models/robotiq/robotiq.g",
        )

        def add_floating_robotiq(prefix, ball_name, pos, color):
            """
            prefix examples:
                "a1_ur_"
                "a2_ur_"
                "h2_a1_ur_"
            """

            # Floating parent ball only
            self.C.addFrame(ball_name) \
                .setParent(self.C.getFrame("world")) \
                .setJoint(ry.JT.free) \
                .setJointState([pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0]) \
                .setShape(ry.ST.sphere, size=[0.08]) \
                .setColor(color) \
                .setContact(0)

            # Import actual Robotiq gripper with original mesh/materials
            self.C.addFile(robotiq_path, namePrefix=prefix)

            gripper_base = f"{prefix}robotiq_base"
            gripper_center = f"{prefix}gripper_center"

            if self.C.getFrame(gripper_base) is None:
                raise RuntimeError(f"Could not find frame: {gripper_base}")

            if self.C.getFrame(gripper_center) is None:
                raise RuntimeError(f"Could not find frame: {gripper_center}")

            # Attach actual gripper to ball.
            # Do NOT setShape() or setColor() on gripper_base.
            self.C.getFrame(gripper_base) \
                .setParent(self.C.getFrame(ball_name)) \
                .setRelativePosition([0.0, 0.0, 0.14]) \
                .setRelativeQuaternion([0.70710678, 0.0, 0.0, 0.70710678])

        # main dual-arm robot: blue
        add_floating_robotiq(
            prefix="a1_ur_",
            ball_name="a1_floating_ball",
            pos=[-0.6, -0.25, 0.8],
            color=[0.0, 0.0, 1.0],
        )

        add_floating_robotiq(
            prefix="a2_ur_",
            ball_name="a2_floating_ball",
            pos=[-0.6, 0.25, 0.8],
            color=[0.0, 0.0, 1.0],
        )

        # support robot: red
        add_floating_robotiq(
            prefix="h1_ur_",
            ball_name="h1_floating_ball",
            pos=[0.8, -2.0, 0.8],
            color=[1.0, 0.0, 0.0],
        )

        add_floating_robotiq(
            prefix="h2_ur_",
            ball_name="h2_floating_ball",
            pos=[0.8, 2.0, 0.8],
            color=[0.0, 1.0, 0.0],
        )


