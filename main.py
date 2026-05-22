from truss import Truss
from backward_search import AssemblyPlanner
from rai.builder import RaiTrussBuilder, RodPathRecord
from DataClasses import AssemblyPlan, RodPathRecord, AttachmentEvent
import numpy as np
import time

def reverse_removal_plan_to_assembly(removal_plan):
    assembly_plan = AssemblyPlan(
        removal_sequence=list(removal_plan.removal_sequence),
        records=[],
    )

    for removal_record in reversed(removal_plan.records):
        rod_id = removal_record.rod_id

        assembly_record = RodPathRecord(rod_id=rod_id)

        # Reverse segment order and reverse each segment internally
        assembly_record.segments = [
            np.asarray(path)[::-1].copy()
            for path in reversed(removal_record.segments)
        ]

        # Removal attaches the rod to the gripper after the "reach installed rod"
        # segment, then carries it to pickup. In reverse, the first segment is
        # the carry back to the scaffold, so placement happens after segment 0.
        place_segment_id = 0 if assembly_record.segments else -1

        # Before first segment:
        # rod starts at pickup pose and is held by the gripper
        assembly_record.events.append(
            AttachmentEvent(
                rod_id=rod_id,
                segment_id=-1,
                parent="a1_ur_gripper_center",
                child=f"rod_{rod_id}",
                action="attach",
            )
        )

        # After the placement segment:
        # rod has reached scaffold pose, so release from gripper
        assembly_record.events.append(
            AttachmentEvent(
                rod_id=rod_id,
                segment_id=place_segment_id,
                parent="a1_ur_gripper_center",
                child=f"rod_{rod_id}",
                action="detach",
            )
        )

        # After the placement segment:
        # rod becomes fixed in scaffold/table
        assembly_record.events.append(
            AttachmentEvent(
                rod_id=rod_id,
                segment_id=place_segment_id,
                parent="table",
                child=f"rod_{rod_id}",
                action="attach",
            )
        )

        assembly_plan.records.append(assembly_record)

    return assembly_plan


run = "replay_husky"
    
if run == "replay_husky":

    truss = Truss.from_json("JSON/scaffold_test.json")

    builder = RaiTrussBuilder(truss, radius=0.005, scale=0.0011)
    builder.import_robots()

    searcher = AssemblyPlanner(truss, builder=builder)

    removal_sequence = searcher.backward_search()
    assembly_sequence = list(reversed(removal_sequence)) if removal_sequence else None

    print("Removal:", removal_sequence)
    print("Assembly:", assembly_sequence)

    if assembly_sequence is None:
        raise RuntimeError("No feasible assembly sequence found")

    recorder = AssemblyPlan()

    # for rod_id in assembly_sequence:
    #     if rod_id not in searcher.motion_records:
    #         raise RuntimeError(f"No recorded motion found for rod {rod_id}")

    #     recorder.add(rod_id, searcher.motion_records[rod_id])

    # print("Recorded assembly sequence:", recorder.sequence())

    # print("Now replaying full recorded plan")

    # replay_builder = RaiTrussBuilder(truss, radius=0.005, scale=0.0011)
    # replay_builder.import_husky()

    # replay_builder.replay_recorded_plan(
    #     recorder,
    #     rod_pos=[-3, -1, 1.0],
    #     rod_ori=[0.5, 0.0, 0.5, 0.70710678],
    #     dt=0.0005,
    #     wait_for_input = True
    # )
    
    

    if not hasattr(searcher, "final_node"):
        raise RuntimeError("Search did not produce a final_node with recorded motions")

    removal_recorder = AssemblyPlan(
        removal_sequence=searcher.final_node.sequence,
        records=searcher.final_node.records,
    )

    assembly_recorder = reverse_removal_plan_to_assembly(removal_recorder)
    
    for record in assembly_recorder.records:
        print("record", record.rod_id)
        for e in record.events:
            print("  event:", e.segment_id, e.action, e.child, "->", e.parent)

    replay_builder = RaiTrussBuilder(truss, radius=0.005, scale=0.0011)
    replay_builder.import_robots()
    
    replay_builder.display_recorded_plan_viser(
        assembly_recorder,
        port=8080,
        pause_time=0.03,
        rod_pos=[-3, -1, 1.0],
        rod_ori=[0.5, 0.0, 0.5, 0.70710678],
        replay_mode="assembly",
    )
    

    print("Replay finished")

else:
    print("Please use a valid name")
