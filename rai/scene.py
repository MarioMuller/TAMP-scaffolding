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

    def import_main_husky(self):
        """
        import the main husky robot with two arms
        
        """

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
             
        q_rotate_90_z = [
            0.70710678,  # w
            0.0,         # x
            0.0,         # y
            0.70710678,  # z
        ]

        # attatch both arms to the husky
        self.C.addFile(robot_path, namePrefix="a1_").setParent(
        self.C.getFrame("husky_coll_right_arm_bulkhead_joint")
            ).setRelativePosition([0, 0, 0]).setRelativeQuaternion(q_rotate_90_z)

        self.C.addFile(robot_path, namePrefix="a2_").setParent(
        self.C.getFrame("husky_coll_left_arm_bulkhead_joint")
            ).setRelativePosition([0, 0, 0]).setRelativeQuaternion(q_rotate_90_z)
        
        # ------------------------------------------------------------------
        # Initialize the 2 x six UR5 joints
        # ------------------------------------------------------------------
            
        # The UR5 joints belong to individual frames.
        arm_joint_names_a1 = [
            joint_name
            for joint_name in self.C.getJointNames()
            if joint_name.startswith("a1_")
        ]
        
        arm_joint_names_a2 = [
            joint_name
            for joint_name in self.C.getJointNames()
            if joint_name.startswith("a2_")
        ]

        if len(arm_joint_names_a1) != 6:
            raise RuntimeError(
                f"Expected 6 joints for {name}, "
                f"found {len(arm_joint_names_a1)}: {arm_joint_names_a1}"
            )

        if len(arm_joint_names_a2) != 6:
            raise RuntimeError(
                f"Expected 6 joints for {name}, "
                f"found {len(arm_joint_names_a2)}: {arm_joint_names_a2}"
            )

        self.C.setJointState(
            [0.0, (-2+0.75)/2, 0.0, 0.0, 0.0, 0.0],
            arm_joint_names_a1,
        )

        self.C.setJointState(
            [0.0, (-2+0.75)/2, 0.0, 0.0, 0.0, 0.0],
            arm_joint_names_a2,
        )

        return
    
    def import_main_husky_baseless(self):
        """
        import the main husky robot with two arms
        
        """

        self._ensure_table()

        # paths to the files
        robot_path = os.path.join(os.path.dirname(__file__), "../src/models/ur5/ur5.g")
        
        # Floating parent ball only
        self.C.addFrame("main_husky_base_1") \
            .setParent(self.C.getFrame("world")) \
            .setJoint(ry.JT.free) \
            .setJointState([0, 0, 0, 1.0, 0.0, 0.0, 0.0]) \
            .setRelativePosition([0, 0, 0.16]) \
            .setShape(ry.ST.sphere, size=[0.08]) \
            .setColor([0, 0, 1]) \
            .setContact(1)
            
        # self.C.addFrame("main_husky_base_1") \
        #     .setParent(self.C.getFrame("world")) \
        #     .setJoint(
        #         ry.JT.transXYPhi,
        #         limits=np.array([-30, 30, -30, 30, -3.14, 3.14])
        #     ) \
        #     .setJointState([0, 0, 0]) \
        #     .setRelativePosition([0, 0, 0.16]) \
        #     .setShape(ry.ST.sphere, size=[0.08]) \
        #     .setColor([0, 0, 1]) \
        #     .setContact(1)


        # attatch both arms to the husky
        self.C.addFile(robot_path, namePrefix="a1_").setParent(
        self.C.getFrame("main_husky_base_1")
            ).setRelativePosition([0, 0, 0]).setRelativeQuaternion([1, 0, 0, 0])

          
            
        # Second arm base is fixed relative to the first base
        self.C.addFrame("main_husky_base_2") \
            .setParent(self.C.getFrame("main_husky_base_1")) \
            .setRelativePosition([0.3, 0, 0]) \
            .setRelativeQuaternion([1.0, 0.0, 0.0, 0.0]) \
            .setShape(ry.ST.sphere, size=[0.08]) \
            .setColor([0, 0, 1]) \
            .setContact(1)     
        
        self.C.addFile(robot_path, namePrefix="a2_").setParent(
        self.C.getFrame("main_husky_base_2")
            ).setRelativePosition([0, 0, 0]).setRelativeQuaternion([1, 0, 0, 0])
        
      
        return
    
    def import_support_husky(
        self,
        name="h2",
        base_q=(3.0, -3.0, 0.0),
        arm="right",  # or "left"
        color=None,
    ):
        """
        Import a support Husky robot with a single arm, positioned away from the main robot.

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
            
        husky_collision_frames = [
            frame
            for frame in self.C.getFrames()
            if (
                frame.name.startswith(husky_prefix)
                and frame.info().get("contact", 0)
            )
        ]

        if color is not None:
            for frame in husky_collision_frames:
                frame.setColor([
                    color[0],
                    color[1],
                    color[2],
                    0.25,
                ])
                
        arm_prefix = f"{name}_a1_"

        # ------------------------------------------------------------------
        # Central UR5 mounting frame
        # ------------------------------------------------------------------

        # The original mounting positions relative to
        # dual_arm_bulkhead_link are:
        #
        # left:  [0.1225,  0.14891, 0.13371]
        # right: [0.1225, -0.14891, 0.13371]
        #
        # Therefore their hardcoded midpoint is:
        #        [0.1225,  0.0,     0.13371]
        # z-offseet of 0.1 is added to adjust height

        center_mount = f"{name}_center_arm_mount"
        mount_parent = f"{husky_prefix}dual_arm_bulkhead_link"

        parent = self.C.getFrame(mount_parent)

        if parent is None:
            raise RuntimeError(
                f"Could not find Husky mounting parent: {mount_parent}"
            )

        self.C.addFrame(center_mount) \
            .setParent(parent) \
            .setRelativePosition([
                0.1225,
                0.0,
                0.19371,
            ]) \
            .setRelativeQuaternion([
                1,
                0.0,
                0.0,
                0.0,
            ])

        # The original centered mounting orientation is approximately
        # -90 degrees around Z. After adding 180 degrees, it becomes
        # +90 degrees:
        #
        # quaternion [w, x, y, z]
        #            [0.7071, 0, 0, 0.7071]

        # ------------------------------------------------------------------
        # Attach the UR5
        # ------------------------------------------------------------------

        self.C.addFile(
            robot_path,
            namePrefix=arm_prefix,
        ) \
            .setParent(self.C.getFrame(center_mount)) \
            .setRelativePosition([0.0, 0.0, 0.0]) \
            .setRelativeQuaternion([1.0, 0.0, 0.0, 0.0])

        # ------------------------------------------------------------------
        # Initialize the six UR5 joints
        # ------------------------------------------------------------------
            
        # The UR5 joints belong to individual frames.
        arm_joint_names = [
            joint_name
            for joint_name in self.C.getJointNames()
            if joint_name.startswith(arm_prefix)
        ]

        if len(arm_joint_names) != 6:
            raise RuntimeError(
                f"Expected 6 joints for {name}, "
                f"found {len(arm_joint_names)}: {arm_joint_names}"
            )

        self.C.setJointState(
            [0.0, (-2+0.75)/2, 0.0, 0.0, 0.0, 0.0],
            arm_joint_names,
        )
        

    def import_pineapple_model(self):
        
        """
        Pineapple model consisting of one main arm (blue) and two support arms (red and green), with floating bases
        
        """

        self._ensure_table()

        robotiq_path = os.path.join(
            os.path.dirname(__file__),
            "../src/models/robotiq/robotiq.g",
        )

        def add_floating_robotiq(prefix, ball_name, pos, color):
            """
            a1, a2 = arm 1 and 2 of main robot
            h1, h2 = support robots:
       
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

        # support robot 1: red
        add_floating_robotiq(
            prefix="h1_ur_",
            ball_name="h1_floating_ball",
            pos=[0.8, -2.0, 0.8],
            color=[1.0, 0.0, 0.0],
        )

        # support robot 2: green
        add_floating_robotiq(
            prefix="h2_ur_",
            ball_name="h2_floating_ball",
            pos=[0.8, 2.0, 0.8],
            color=[0.0, 1.0, 0.0],
        )


