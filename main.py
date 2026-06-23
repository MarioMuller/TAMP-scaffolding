from truss import Truss
from backward_search import AssemblyPlanner
from rai.builder import RaiTrussBuilder
from DataClasses import AssemblyPlan
import numpy as np

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


if not hasattr(searcher, "final_node"):
    raise RuntimeError("Search did not produce a final_node with recorded motions")

removal_recorder = AssemblyPlan(
    removal_sequence=searcher.final_node.sequence,
    records=searcher.final_node.records,
)


assembly_recorder = AssemblyPlan.reverse_removal_plan_to_assembly(removal_recorder)

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

