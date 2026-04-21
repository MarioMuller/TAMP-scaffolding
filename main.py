from json_import import Truss
from Simple_search import TrussSearch
from RAI_scaffold_sample import RaiTrussBuilder
import time

run = "husky"

if run == "ur5":

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

elif run == "husky":
    # run the backward search to find the assembly sequence
    # truss = Truss.from_json("JSON/long_beam_test.json")
    truss = Truss.from_json("JSON/scaffold_test.json")
    searcher = TrussSearch(truss)

    removal_sequence = searcher.backward_search()
    assembly_sequence = list(reversed(removal_sequence)) if removal_sequence else None
    print("Assembly:", assembly_sequence)

    # Create start environment
    builder = RaiTrussBuilder(truss, radius=0.005, scale=0.001) #scale = 0.0005
    # builder = RaiTrussBuilder(truss, radius=0.005, scale=0.003) #long_beam 0.003 - 0.004
    builder.import_husky()

    # Loop over the assembly_sequence and execute the required steps to build Truss
    first_rod = True
    for rod_id in assembly_sequence:

        builder.create_rod(rod_id, pos=[-3, -1, 1.0], ori= [0.5, 0.0, 0.5, 0.70710678])

        
        keyframes, q0 = builder.get_keyframes(rod_id)
        builder.find_path(keyframes, q0, rod_id)
        
        # keyframes, q0 = builder.husky_direct_komo(rod_id)
        
        # builder.set_to_end_position(rod_id)

    print("Finished all rods palced in desired target location")

else:
    print("Please use a valid name")