from json_import import Truss
from Simple_search import TrussSearch
from RAI_scaffold_sample import RaiTrussBuilder
import time

run = "husky"
# run = "collision test"

if run == "husky":
    # run the backward search to find the assembly sequence
    # truss = Truss.from_json("JSON/long_beam_test.json")
    truss = Truss.from_json("JSON/scaffold_test.json")
    searcher = TrussSearch(truss)

    removal_sequence = searcher.backward_search()
    assembly_sequence = list(reversed(removal_sequence)) if removal_sequence else None
    
    print("Assembly:", assembly_sequence)

    # Create start environment
    builder = RaiTrussBuilder(truss, radius=0.005, scale=0.0006) #scale = 0.0005
    # builder = RaiTrussBuilder(truss, radius=0.005, scale=0.003) #long_beam 0.003 - 0.004
    builder.import_husky()

    # Loop over the assembly_sequence and execute the required steps to build Truss
    first_rod = True
    for rod_id in assembly_sequence:

        builder.create_rod(rod_id, pos=[-3, -1, 1.0], ori= [0.5, 0.0, 0.5, 0.70710678])
        
        keyframes, q0 = builder.get_keyframes(rod_id)
        # builder.find_path(keyframes, q0, rod_id, show_visualization=True)
        builder.find_path_shortcut(keyframes, q0, rod_id, do_shortcut=True)
       
        
        # keyframes, q0 = builder.husky_direct_komo(rod_id)
        
        # builder.set_to_end_position(rod_id)

    print("Finished all rods palced in desired target location")
    time.sleep(5)
    
elif run == "collision test":
    
    truss = Truss.from_json("JSON/scaffold_test.json")
    searcher = TrussSearch(truss)

    removal_sequence = searcher.backward_search()
    assembly_sequence = list(reversed(removal_sequence)) if removal_sequence else None
    
    # Create start environment
    builder = RaiTrussBuilder(truss, radius=0.005, scale=0.001) #scale = 0.0005
    builder.import_husky()
    
    builder.create_rod(assembly_sequence[1], pos=[-1, -0, 0.05], ori= [0.7071,0.7071,0,0])
    builder.super_simple_collision()
    time.sleep(5)

    print("collision test finsihed")

else:
    print("Please use a valid name")