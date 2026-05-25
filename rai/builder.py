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
        
    def import_floating_grippers_debug(self):
        self.scene.import_floating_grippers_debug()
        
    def import_robots(self):
        
        debug = True
        if debug:
            self.import_floating_grippers_debug()
            
        else: 
            self.import_husky()

            self.import_support_husky(
                name="h2",
            )
        
    def display_recorded_plan_viser(self, *args, **kwargs):
        return self.viser_replayer.display_recorded_plan_viser(*args, **kwargs)
    
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
            
            
    def try_remove_and_commit_rod(
        self,
        current_state,
        new_state,
        rod_id,
        q_start=None,
        supported=None,
        support_required=False,
        use_rrt=False,
        do_shortcut=False,
    ):
        """
        Backward-search motion test.

        current_state:
            rods currently installed before candidate removal

        new_state:
            rods remaining after candidate rod is removed

        supported:
            dict support_gripper -> rod_id
        """

        if supported is None:
            supported = {}

        # 1. Build scene with candidate rod still installed
        self.reset_scene_with_rods(current_state)

        if q_start is not None:
            self.C.setJointState(q_start)

        # 2. Restore branch-local support attachments
        for support_gripper, supported_rod in supported.items():
            print("Support robot is actually used")
            rod_frame = f"rod_{supported_rod}"
            if self.C.getFrame(rod_frame) is not None:
                self.C.attach(support_gripper, rod_frame)

        record = RodPathRecord(rod_id=rod_id)

        candidate_is_supported = rod_id in supported.values()

        # 3. If candidate is currently supported, remember which gripper holds it
        old_support_gripper = None
        for gripper, supported_rod in supported.items():
            if supported_rod == rod_id:
                old_support_gripper = gripper
                break

        # 4. Compute removal keyframes
        if False:
            print("error")
        # if support_required:
        #     keyframes, q0, new_supported = self.keyframes.get_remove_keyframes_with_support(
        #         rod_id=rod_id,
        #         supported=supported,
        #         candidate_is_supported=candidate_is_supported,
        #         old_support_gripper=old_support_gripper,
        #     )
        else:
            keyframes, q0, new_supported = self.keyframes.get_remove_keyframes_dual(
                rod_id=rod_id,
                supported=supported,
                candidate_is_supported=candidate_is_supported,
                old_support_gripper=old_support_gripper,
            )

        if keyframes is None:
            return None

        # 5. Convert keyframes into path segments
        q_current = self.C.getJointState().copy()
        for i, q_goal in enumerate(keyframes):
            if use_rrt:
                path = self.paths.plan_segment(
                    q_start=q_current,
                    q_goal=q_goal,
                    do_shortcut=do_shortcut,
                )
            else:
                path = np.asarray([q_current, q_goal])

            if path is None:
                return None

            record.segments.append(path)

            self.C.setJointState(q_goal)
            q_current = q_goal.copy()

            if i == 0:
                # Bookkeeping: rod is no longer attached to scaffold/table.
                record.events.append(
                    AttachmentEvent(
                        rod_id=rod_id,
                        segment_id=i,
                        parent="table",
                        child=f"rod_{rod_id}",
                        action="detach",
                    )
                )

                print(f"[event] segment={i}: detach rod_{rod_id} from table")

                # Actual RAI operation: re-parent rod to gripper.
                self._attach_and_record(
                    record=record,
                    rod_id=rod_id,
                    segment_id=i,
                    parent="a1_ur_gripper_center",
                    child=f"rod_{rod_id}",
                )

        return {
            "record": record,
            "q_final": q_current,
            "supported": new_supported,
        }
               
   

if __name__ == "__main__":

    truss = Truss.from_json("JSON/long_beam_test.json")

    # build_entire_truss_in_rai(radius, node_positions, rods, C)
    builder = RaiTrussBuilder(truss, radius=0.0015)
   # builder.import_robots()
