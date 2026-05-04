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
    
    