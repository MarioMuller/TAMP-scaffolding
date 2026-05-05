from truss import Truss
import numpy as np
from DataClasses import RodPathRecord, AttachmentEvent
from .scene import RaiScene
from .rods import RodManager
from .keyframes import KeyframePlanner
from .pathplanning import PathPlanner
from .replay import PlanReplayer
from .viser_replay import ViserPlanReplayer


class RaiTrussBuilder:

    def __init__(self, truss, radius=0.0015, scale = 0.00351):
        self.truss = truss
        self.radius = radius
        self.scale = scale 
        
        self.scene = RaiScene()
        self.C = self.scene.C
        
        self.rods = RodManager(self.C, truss, radius=radius, scale=scale)
        
        self.keyframes = KeyframePlanner(self.C, self.rods)
        self.paths = PathPlanner(self.C)
        self.replayer = PlanReplayer(self.C, self.rods)
        self.viser_replayer = ViserPlanReplayer(self.C, self.rods)

    def import_husky(self):
        self.scene.import_husky()
    
    
    def replay_recorded_plan(self, *args, **kwargs):
        return self.replayer.replay_recorded_plan(*args, **kwargs)
    
    def display_recorded_plan_viser(self, *args, **kwargs):
        return self.viser_replayer.display_recorded_plan_viser(*args, **kwargs)
                        
    def try_plan_and_commit_rod(
        self,
        rod_id,
        rod_pos=[-3, -1, 1.0],
        rod_ori=[0.5, 0.0, 0.5, 0.70710678],
        do_shortcut=True,
        shortcut_iter=300,
        shortcut_step=0.02,
        replay_now=False,
    ):
        """
        Checks if path for placement can be found if this rod is removed
        """

        print(f"\nTrying rod {rod_id}")

        try:
            self.rods.create_rod(rod_id, pos=rod_pos, ori=rod_ori)

            # keyframes, q0 = self.keyframes.get_keyframes(rod_id)
            
            keyframes, q0 = self.keyframes.get_keyframes_dual(rod_id)

            record = RodPathRecord(rod_id=rod_id)

            q_start = np.asarray(q0, dtype=float).copy()

            for keyframe_id, q_goal in enumerate(keyframes):
                q_goal = np.asarray(q_goal, dtype=float).copy()
                self.C.setJointState(q_start)

                path = None

                path = self.paths.plan_segment(
                    q_start=q_start,
                    q_goal=q_goal,
                    do_shortcut=do_shortcut,
                    shortcut_iter=shortcut_iter,
                    shortcut_step=shortcut_step,
                    rrt_attempts=50,
                )

                record.segments.append(path.copy())

                if replay_now:
                    self.paths.play_path(path, dt=0.005, title=f"rod {rod_id}, segment {keyframe_id}")

                # Commit scene mode after reaching keyframe
                self.C.setJointState(q_goal)

                if keyframe_id == 0:
                    self.C.attach("a1_ur_gripper_center", f"rod_{rod_id}")
                    record.events.append(
                        AttachmentEvent(
                            rod_id=rod_id,
                            segment_id=keyframe_id,
                            parent="a1_ur_gripper_center",
                            child=f"rod_{rod_id}",
                        )
                    )
                    print(f"Rod {rod_id} attached to robot")

                elif keyframe_id == 1:
                    self.C.attach("table", f"rod_{rod_id}")
                    record.events.append(
                        AttachmentEvent(
                            rod_id=rod_id,
                            segment_id=keyframe_id,
                            parent="table",
                            child=f"rod_{rod_id}",
                        )
                    )
                    print(f"Rod {rod_id} attached to table")

                q_start = q_goal.copy()

            print(f"Accepted rod {rod_id}")
            return record

        except Exception as e:
            print(f"Rod {rod_id} failed: {e}")
            return None
    
    def reset_scene_with_rods(self, placed_rods):
        """
        Rebuilds scene with not yet removed rods in final pos
        """

        self.C.clear()
        self.C.addFrame("world")
        self.import_husky()

        for rod_id in placed_rods:
            self.rods.create_rod(
                rod_id,
                pos=[-3, -1, 1.0],
                ori=[0.5, 0.0, 0.5, 0.70710678],
            )

            self.rods.set_to_goal_pose(rod_id, view=False)

            self.C.attach("table", f"rod_{rod_id}")
            
        self.C.view()
        


if __name__ == "__main__":

    truss = Truss.from_json("JSON/long_beam_test.json")

    # build_entire_truss_in_rai(radius, node_positions, rods, C)
    builder = RaiTrussBuilder(truss, radius=0.0015)
    builder.import_husky()
