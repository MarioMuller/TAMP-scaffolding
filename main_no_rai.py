from truss import Truss
from backward_search import AssemblyPlanner
from rigidityCheck.structural_replay import display_structural_assembly


truss = Truss.from_json(
    "JSON/own_examples/260724_stability_ini.json"
)

searcher = AssemblyPlanner(
    truss=truss,
    builder=None,
    max_supports=2,
)

removal_sequence = searcher.backward_search()

if removal_sequence is None:
    raise RuntimeError("No structurally feasible sequence found.")

assembly_sequence = list(reversed(removal_sequence))

print("Removal:", removal_sequence)
print("Assembly:", assembly_sequence)

display_structural_assembly(
    truss=truss,
    removal_steps=searcher.final_node.structural_steps,
    scale=0.0011,
    label_rods=True,
    video_path="structural_assembly.mp4",
    seconds_per_step=0.8,
    fps=30,
)