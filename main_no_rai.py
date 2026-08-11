from truss import Truss
from backward_search import AssemblyPlanner
from rigidityCheck.structural_replay import display_structural_assembly
import time
import pickle
from pathlib import Path


truss = Truss.from_json(
    # "JSON/own_examples/260724_stability_ini.json"
    # "JSON/own_examples/diy_proper_full.json"
    # "JSON/own_examples/260804_FoC_demo.json"
    "JSON/own_examples/260804_RobArchDemo_ini.json"
)

 # Filter to chose a subset of rods to include in the search. This is useful for testing
selected_rods = {
    0, 1, 2, 5, 6, 7, 8, 9, 13, 15
    # 45, 33, 55, 53, 47, 35, 50, 46, 34, 49, 38, 51, 39, 52, 54
}

unknown_rods = selected_rods - set(truss.elements)
if unknown_rods:
    raise ValueError(
        f"Selected rods do not exist: {sorted(unknown_rods)}"
    )

# Keep only the selected rods.
truss.elements = {
    rod_id: endpoints
    for rod_id, endpoints in truss.elements.items()
    if rod_id in selected_rods
}

# Keep grounding only for selected rods.
truss.grounded_rods &= selected_rods

# Keep couplers only when both connected rods are selected.
truss.couplers = {
    (rod_1, rod_2)
    for rod_1, rod_2 in truss.couplers
    if rod_1 in selected_rods and rod_2 in selected_rods
}

print("Included rods:", sorted(truss.elements))

searcher = AssemblyPlanner(
    truss=truss,
    builder=None,
    max_supports=2,
)

start_time = time.time()
removal_sequence = searcher.backward_search(capture_key="v",)
end_time = time.time()
print(f"Backward search took {end_time - start_time:.2f} seconds.")

if removal_sequence is None:
    raise RuntimeError("No structurally feasible sequence found.")

assembly_sequence = list(reversed(removal_sequence))

# Save before printing or displaying.
capture_path = Path("debug_captures/structural_sequence.pkl")
capture_path.parent.mkdir(parents=True, exist_ok=True)

with capture_path.open("wb") as file:
    pickle.dump(
        {
            "truss": truss,
            "removal_sequence": removal_sequence,
            "assembly_sequence": assembly_sequence,
            "structural_steps": searcher.final_node.structural_steps,
        },
        file,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

print(f"Saved structural configurations to: {capture_path}")

print("Removal:", removal_sequence)
print("Assembly:", assembly_sequence)

display_structural_assembly(
    truss=truss,
    removal_steps=searcher.final_node.structural_steps,
    scale=0.0011,
    label_rods=True,
    # video_path="FoC_demo_assembly.mp4",
    video_path = None,
    seconds_per_step=0.8,
    fps=30,
)