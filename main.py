from json_import import Truss
from Simple_search import TrussSearch
from RAI_scaffold_sample import RaiTrussBuilder
import time

# run the backward search to find the assembly sequence
truss = Truss.from_json("JSON/long_beam_test.json")
searcher = TrussSearch(truss)

removal_sequence = searcher.backward_search()
assembly_sequence = list(reversed(removal_sequence)) if removal_sequence else None
print("Assembly:", assembly_sequence)

# Create start environment
builder = RaiTrussBuilder(truss, radius=0.003)
builder.import_ur5()

# Loop over the assembly_sequence and execute the required steps to build Truss
first_rod = True
for rod_id in assembly_sequence:

    builder.create_rod(rod_id)

    if not first_rod:
        builder.prepare_next_grab(rod_id)
        
    first_rod = False
    builder.pick_and_place_rod(rod_id)


print("Finished all rods palced in desired target location")