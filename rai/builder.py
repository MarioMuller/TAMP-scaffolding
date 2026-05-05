from truss import Truss
import numpy as np
from DataClasses import RodPathRecord, AttachmentEvent
from .scene import RaiScene
from .rods import RodManager
from .keyframes import KeyframePlanner
from .pathplanning import PathPlanner
from .replay import PlanReplayer
from .viser_replay import ViserPlanReplayer
import time

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
        
    def import_support_husky(self, name="h2", base_q=(3.0, -3.0, 0.0)):
        self.scene.import_support_husky(
            name=name,
            base_q=base_q,
        )
        
    def import_robots(self):
        self.import_husky()

        self.import_support_husky(
            name="h2",
        )
    
    
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
        use_rrt = True,
        needs_support=False,
        release_supported_rod=None,
        support_gripper="h2_a1_ur_gripper_center",
    ):
        """
        Checks if path for placement can be found if this rod is removed
        """

        print(f"\nTrying rod {rod_id}")

        try:
            self.rods.create_rod(rod_id, pos=rod_pos, ori=rod_ori)

            keyframes, q0 = self.keyframes.get_keyframes(rod_id)
            
            # keyframes, q0 = self.keyframes.get_keyframes_dual(
            #     rod_id,
            #     d1_from_end=0.04,
            #     d12_between_arms=0.12,
            # )

            record = RodPathRecord(rod_id=rod_id)

            q_start = np.asarray(q0, dtype=float).copy()

            for keyframe_id, q_goal in enumerate(keyframes):
                q_goal = np.asarray(q_goal, dtype=float).copy()
                self.C.setJointState(q_start)

                path = None

                if use_rrt:
                    path = self.paths.plan_segment(
                        q_start=q_start,
                        q_goal=q_goal,
                        do_shortcut=do_shortcut,
                        shortcut_iter=shortcut_iter,
                        shortcut_step=shortcut_step,
                        rrt_attempts=50,
                    )

                    if path is None:
                        raise RuntimeError(
                            f"RRT failed for rod {rod_id}, segment {keyframe_id}"
                        )
                else:
                    path = np.asarray([q_goal], dtype=float)

                record.segments.append(path.copy())

                if replay_now:
                    self.paths.play_path(path, dt=0.005, title=f"rod {rod_id}, segment {keyframe_id}")

                # Commit scene mode after reaching keyframe
                self.C.setJointState(q_goal)

                if keyframe_id == 0:
                    self._attach_and_record(
                        record=record,
                        rod_id=rod_id,
                        segment_id=keyframe_id,
                        parent="a1_ur_gripper_center",
                        child=f"rod_{rod_id}",
                    )

                elif keyframe_id == 1:
                    # Previous rod becomes stable once this rod is placed.
                    if release_supported_rod is not None:
                        self._attach_and_record(
                            record=record,
                            rod_id=release_supported_rod,
                            segment_id=keyframe_id,
                            parent="table",
                            child=f"rod_{release_supported_rod}",
                        )

                        print(f"Released supported rod {release_supported_rod} to table")

                    # Current rod is unstable and needs support.
                    if needs_support:
                        self.move_support_to_rod_and_attach(
                            record=record,
                            rod_id=rod_id,
                            support_gripper=support_gripper,
                            use_rrt=False,
                            shortcut_step=shortcut_step,
                        )
                    else:
                        self._attach_and_record(
                            record=record,
                            rod_id=rod_id,
                            segment_id=keyframe_id,
                            parent="table",
                            child=f"rod_{rod_id}",
                        )

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
        self.import_robots()

        for rod_id in placed_rods:
            self.rods.create_rod(
                rod_id,
                pos=[-3, -1, 1.0],
                ori=[0.5, 0.0, 0.5, 0.70710678],
            )

            self.rods.set_to_goal_pose(rod_id, view=False)

            self.C.attach("table", f"rod_{rod_id}")
            
        self.C.view()
    
    
    def _attach_and_record(
        self,
        record,
        rod_id,
        segment_id,
        parent,
        child,
    ):
        self.C.attach(parent, child)

        record.events.append(
            AttachmentEvent(
                rod_id=rod_id,
                segment_id=segment_id,
                parent=parent,
                child=child,
                action="attach",
            )
        )

        print(f"[event] segment={segment_id}: {child} -> {parent}")
        
        
    def move_support_to_rod_and_attach(
        self,
        record,
        rod_id,
        support_gripper="h2_a1_ur_gripper_center",
        main_gripper="a1_ur_gripper_center",
        use_rrt=False,
        shortcut_step=0.02,
    ):
        """
        Sequential support step:
        - Main robot is assumed to hold the rod at the target.
        - Main gripper is frozen inside support KOMO.
        - Support robot moves to the rod.
        - Rod is transferred to support gripper.
        """

        support_keyframes, support_q0 = self.keyframes.get_support_keyframes(
            rod_id,
            support_gripper=support_gripper,
            main_gripper=main_gripper,
            freeze_main=False,
            keep_rod_at_target=True,
        )

        q_start = np.asarray(support_q0, dtype=float).copy()
        q_goal = np.asarray(support_keyframes[-1], dtype=float).copy()

        if use_rrt:
            path = self.paths.plan_segment(
                q_start=q_start,
                q_goal=q_goal,
                do_shortcut=True,
                shortcut_iter=300,
                shortcut_step=shortcut_step,
                rrt_attempts=50,
            )

            if path is None:
                raise RuntimeError(f"Support robot failed to reach rod {rod_id}")
        else:
            path = self.paths.interpolate_path(
                np.asarray([q_start, q_goal], dtype=float),
                max_step=shortcut_step,
            )

        record.segments.append(path.copy())
        support_segment_id = len(record.segments) - 1

        self.C.setJointState(q_goal)

        self._attach_and_record(
            record=record,
            rod_id=rod_id,
            segment_id=support_segment_id,
            parent=support_gripper,
            child=f"rod_{rod_id}",
        )

        print(f"Support robot now holds rod {rod_id}")
        
    def show_keyframes(self, keyframes, title="keyframe", dt=1.0):
        """
        Visualize a list/array of keyframes in the RAI viewer.
        """
        if keyframes is None:
            print("No keyframes to show")
            return

        for i, q in enumerate(keyframes):
            print(f"Showing {title} {i}/{len(keyframes) - 1}")
            self.C.setJointState(q)
            self.C.view(False, f"{title} {i}")
            time.sleep(dt)


if __name__ == "__main__":

    truss = Truss.from_json("JSON/long_beam_test.json")

    # build_entire_truss_in_rai(radius, node_positions, rods, C)
    builder = RaiTrussBuilder(truss, radius=0.0015)
    self.import_robots()
