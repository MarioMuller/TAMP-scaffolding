from json_import import Truss
from Simple_search import TrussSearch
from json_to_rai_conversion import RaiTrussBuilder

# run the backward search to find the assembly sequence
truss = Truss.from_json("JSON/long_beam_test.json")
searcher = TrussSearch(truss)

removal_sequence = searcher.backward_search()
assembly_sequence = list(reversed(removal_sequence)) if removal_sequence else None
print("Assembly:", assembly_sequence)

# Create start environment
builder = RaiTrussBuilder(truss)
builder.import_panda()

# Loop over the assembly_sequence and execute the required steps to build Truss
for rod_id in assembly_sequence:

    # rod_id = 14
    builder.create_rod(rod_id)
    # builder.show_target(rod_id)
    # builder.grab_rod(rod_id)

    builder.place_rod(rod_id)
    # builder.pick_and_place_rod(rod_id)

print(rod_id)



    # builder.C.view()
    # input("Press Enter to close...")
    