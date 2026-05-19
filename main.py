from truss import Truss
from backward_search import AssemblyPlanner
from rai.builder import RaiTrussBuilder, RodPathRecord
from DataClasses import AssemblyPlan
import time

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
    
    

    for rod_id in removal_sequence:
        if rod_id not in searcher.motion_records:
            raise RuntimeError(f"No recorded motion found for rod {rod_id}")

        recorder.add(rod_id, searcher.motion_records[rod_id])

    replay_builder = RaiTrussBuilder(truss, radius=0.005, scale=0.0011)
    replay_builder.import_robots()
    
    replay_builder.display_recorded_plan_viser(
        recorder,
        port=8080,
        pause_time=0.03,
        rod_pos=[-3, -1, 1.0],
        rod_ori=[0.5, 0.0, 0.5, 0.70710678],
    )
    
    

    print("Replay finished")

else:
    print("Please use a valid name")