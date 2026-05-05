import os
import numpy as np
import robotic as ry


class RaiScene:
    def __init__(self):
        self.C = ry.Config()
        self.C.addFrame("world")

    def clear(self):
        self.C.clear()
        self.C.addFrame("world")
        
    def import_husky(self):

        # sets ground plane
        table = self.C.addFrame("table").setPosition([0, 0, 0.0]).setShape(
            ry.ST.box, size=[20, 20, 0.02, 0.005]
        ).setColor([0.9, 0.9, 0.9]).setContact(1)

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

        # ensure table exists
        if self.C.getFrame("table") is None:
            self.C.addFrame("table") \
                .setPosition([0, 0, 0.0]) \
                .setShape(ry.ST.box, size=[20, 20, 0.02, 0.005]) \
                .setColor([0.9, 0.9, 0.9]) \
                .setContact(1)

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
    