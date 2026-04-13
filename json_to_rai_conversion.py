from json_import import Truss
import numpy as np
import robotic as ry
import time

class RaiTrussBuilder:

    def __init__(self, truss, radius=0.0015):
        self.truss = truss
        self.radius = radius
        self.C = ry.Config()
        self.C.addFrame("world")
        self.scale = 0.00351

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
                return np.array([1.0, 0.0, 0.0, 0.0])
            else:
                return np.array([0.0, 1.0, 0.0, 0.0])

        axis = axis / axis_norm
        angle = np.arccos(np.clip(dot, -1.0, 1.0))

        w = np.cos(angle / 2.0)
        xyz = axis * np.sin(angle / 2.0)

        return np.concatenate(([w], xyz))

    # only here as a helper to see that the calculations work properly and lead to desired final structure
    def check_end_configuration(self):
        
        for rod_id, (n1, n2) in self.truss.elements.items():

            p1 = np.array(self.truss.nodes[n1]) * self.scale
            p2 = np.array(self.truss.nodes[n2]) * self.scale
        
            length = np.linalg.norm(p2 - p1)
            center = 0.5 * (p1 + p2)
            quat = self.quaternion_from_z_to_vector(p2 - p1)

            self.C.addFrame(f"rod_{rod_id}", 'world') .setShape(ry.ST.cylinder, [length, self.radius]) .setColor([.5,1.,.0]) .setPosition(center) .setQuaternion(quat)
        
        self.C.view()
        input("Press Enter to close...")

        return
    
    def get_goal_pose(self, rod_id):

        n1, n2 = self.truss.elements[rod_id]

        p1 = np.array(self.truss.nodes[n1]) * self.scale
        p2 = np.array(self.truss.nodes[n2]) * self.scale

        center = 0.5 * (p1 + p2)
        center = center + [0, 0, 0.1]
        quat = self.quaternion_from_z_to_vector(p2 - p1)

        return center, quat

    # creates the next required rod
    def create_rod(self, rod_id):

        n1, n2 = self.truss.elements[rod_id]

        p1 = np.array(self.truss.nodes[n1], dtype=float) * self.scale
        p2 = np.array(self.truss.nodes[n2], dtype=float) * self.scale

        length = np.linalg.norm(p2 - p1) -0.02
        if length < 1e-10:
            raise ValueError(f"Rod {rod_id} has zero length")

        self.C.addFrame(f"rod_{rod_id}") .setShape(ry.ST.cylinder, [length, self.radius]) .setColor([.5,1.,.0]) .setPosition([0.1,-0.4,0.1]) .setQuaternion([0.7070, 1, 0, 0.7070]) #.setContact(1)
        self.C.view()
        # input("Press Enter to close...")
        return
    
    def import_ur5(self):
        self.C.addFile("/home/mario/TAMP-scaffolding/src/models/ur5/ur5.g") .setPosition([0.5,0,0])
        print(self.C.getFrameNames())

        self.qHome = self.C.getJointState().copy()
        self.C.view()
        return  

    def show_target(self, rod_id):
        goal_center, goal_quat = self.get_goal_pose(rod_id)

        name = f"rod_{rod_id}_target"
        if self.C.getFrame(name) is None:
            self.C.addFrame(name, "world")

        self.C.getFrame(name).setShape(ry.ST.marker, [0.1])
        self.C.getFrame(name).setColor([1, 0, 0])
        self.C.getFrame(name).setPosition(goal_center)
        self.C.getFrame(name).setQuaternion(goal_quat)

        self.C.view()

        return

    
    def pick_and_place_rod(self, rod_id):

        goal_center, goal_quat = self.get_goal_pose(rod_id)

        target_name = f"rod_{rod_id}_target"
        if self.C.getFrame(target_name) is None:
            self.C.addFrame(target_name, 'world')

        self.C.getFrame(target_name).setPosition(goal_center)
        self.C.getFrame(target_name).setQuaternion(goal_quat)
        
        komo = ry.KOMO(self.C, phases=2, slicesPerPhase=10, kOrder=2, enableCollisions=True)

        komo.addControlObjective([], 0, 1e-1) # what happens if you change weighting to 1e0? why?
        komo.addControlObjective([], 2, 1e0)

        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])

        # grab the rod in spawning position
        komo.addObjective([1.], ry.FS.positionDiff, ['ur_gripper_center', f"rod_{rod_id}"], ry.OT.eq, [1e1]) # an equality constraint on the 3D position difference between ur_gripper_center and box
        komo.addObjective([1.], ry.FS.scalarProductXZ, ['ur_gripper_center', f"rod_{rod_id}"], ry.OT.eq, [1e1], [1.0])
        komo.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e0], [], 1)
        komo.addModeSwitch([1.,-1], ry.SY.stable, ['ur_gripper_center', f"rod_{rod_id}"], True)

        komo.addObjective([2.], ry.FS.positionDiff, [f"rod_{rod_id}", target_name], ry.OT.eq, [1e1])
        komo.addObjective([2.], ry.FS.qItself, [], ry.OT.eq, [1e0], [], 1)
        komo.addObjective([2.], ry.FS.scalarProductZZ, [f"rod_{rod_id}", target_name], ry.OT.eq, [1e1], [1.0])

        ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
        print(ret)

        if not ret.feasible:
            komo.view(True, "IK solution")
            raise RuntimeError(f"Pick & PLace not possible for rod {rod_id}")

        # komo.view(True, "IK solution")
        q = komo.getPath()

        print('size of path:', q.shape)

        for t in range(q.shape[0]):
            if t == 10:
                self.C.attach('ur_gripper_center', f'rod_{rod_id}')

            self.C.setJointState(q[t])
            self.C.view(False, f'place waypoint {t}')
            time.sleep(.1)

        self.C.attach('world', f'rod_{rod_id}')

        return   
    
    def prepare_next_grab(self, rod_id):

        print("prepare next grab")
        
        komo = ry.KOMO(self.C, phases=2, slicesPerPhase=10, kOrder=2, enableCollisions=True)

        komo.addControlObjective([], 0, 1e-1) # what happens if you change weighting to 1e0? why?
        komo.addControlObjective([], 2, 1e0)

        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])

        tool_pos = np.array(self.C.getFrame('ur_gripper_center').getPosition())
        tool_pos_up = tool_pos.copy()
        tool_pos_up[2] += 0.20   # move 10 cm up in world z

        komo.addObjective([1.], ry.FS.position, ['ur_gripper_center'], ry.OT.eq, [1e1], tool_pos_up)

        rod_center = np.array(self.C.getFrame(f"rod_{rod_id}").getPosition())
        rod_center_up = rod_center.copy()
        rod_center_up[2] += 0.10   # move 20 cm up in world z
        komo.addObjective([2.], ry.FS.position, ['ur_gripper_center'], ry.OT.eq, [1e1], rod_center_up)
        komo.addObjective([2.], ry.FS.scalarProductXZ, ['ur_gripper_center', f"rod_{rod_id}"], ry.OT.eq, [1e1], [1.0])
        komo.addObjective([2.], ry.FS.scalarProductZZ, ['ur_gripper', 'world'], ry.OT.eq,[1e1],[-1.])

        ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
        print(ret)

        if not ret.feasible:
            komo.view(True, "IK solution")
            raise RuntimeError(f"Pick & PLace not possible for rod {rod_id}")

        # komo.view(True, "IK solution")
        q = komo.getPath()

        print('size of path:', q.shape)

        for t in range(q.shape[0]):
            self.C.setJointState(q[t])
            self.C.view(False, f'place waypoint {t}')
            time.sleep(.1)

        return   
    

    # def grab_rod(self, rod_id):

    #     # current assumption the rod is always spawned in the same starting location
    #     komo = ry.KOMO(self.C, phases=1, slicesPerPhase=10, kOrder=2, enableCollisions=False)

    #     komo.addControlObjective([], 0, 1e-1) # what happens if you change weighting to 1e0? why?
    #     komo.addControlObjective([], 2, 1e0)

    #     komo.addObjective([1.], ry.FS.positionDiff, ['ur_gripper_center', f"rod_{rod_id}"], ry.OT.eq, [1e1]) # an equality constraint on the 3D position difference between gripper and box
    #     komo.addObjective([1.], ry.FS.scalarProductYZ, ['ur_gripper_center', f"rod_{rod_id}"], ry.OT.eq, [1e1], [1.0])
    #     komo.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e0], [], 1)

    #     # komo.addRigidSwitch(1.0, ['ur_gripper_center', f"rod_{rod_id}"])

    #     ret = ry.NLP_Solver(komo.nlp(), verbose=0) .solve()
    #     print(ret)

    #     if not ret.feasible:
    #         raise RuntimeError(f"Grabbing not possible for rod {rod_id}")

    #     q = komo.getPath()
    #     print('size of path:', q.shape)

    #     for t in range(q.shape[0]):
    #         self.C.setJointState(q[t])
    #         self.C.view(False, f'waypoint {t}')
    #         time.sleep(.1)
        
    #     self.C.attach('ur_gripper_center', f'rod_{rod_id}')

    #     return

    # def place_rod(self, rod_id):
        
    #     q_home = self.C.getJointState()

    #     goal_center, goal_quat = self.get_goal_pose(rod_id)

    #     target_name = f"rod_{rod_id}_target"
    #     if self.C.getFrame(target_name) is None:
    #         self.C.addFrame(target_name, 'world')

    #     self.C.getFrame(target_name).setPosition(goal_center)
    #     self.C.getFrame(target_name).setQuaternion(goal_quat)

    #     komo = ry.KOMO(self.C, phases=1, slicesPerPhase=10, kOrder=2, enableCollisions=False)
        
    #     komo.addControlObjective([], 0, 1e-1) # what happens if you change weighting to 1e0? why?
    #     komo.addControlObjective([], 2, 1e0)

    #     komo.addObjective([1.], ry.FS.positionDiff, [f"rod_{rod_id}", target_name], ry.OT.eq, [1e1])
    #     komo.addObjective([1.], ry.FS.quaternionDiff, [f"rod_{rod_id}", target_name], ry.OT.eq, [1e1])

    #     ret = ry.NLP_Solver(komo.nlp(), verbose=0) .solve()
    #     print(ret)

    #     if not ret.feasible:
    #         raise RuntimeError(f"Placemet not possible for rod {rod_id}")

    #     q = komo.getPath()
    #     print('size of path:', q.shape)

    #     for t in range(q.shape[0]):
    #         self.C.setJointState(q[t])
    #         self.C.view(False, f'place waypoint {t}')
    #         time.sleep(.1)

    #     # self.C.setJointState(q_home)
    #     self.C.attach('world', f'rod_{rod_id}')
    #     self.C.view()
    #     time.sleep(1.)

    #     return

if __name__ == "__main__":

    truss = Truss.from_json("JSON/long_beam_test.json")

    # build_entire_truss_in_rai(radius, node_positions, rods, C)
    builder = RaiTrussBuilder(truss, radius=0.0015)
    builder.import_ur10()
    builder.create_rod(4)
    builder.check_end_configuration()
    builder.grab_rod(3)
    