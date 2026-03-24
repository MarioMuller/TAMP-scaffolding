import json
from collections import defaultdict, deque
import pyvista as pv
import numpy as np
import heapq

# import data from json file
with open("JSON/long_beam_test.json", "r") as d:
    data = json.load(d)

# feed nodes to dict and create set of grounded nodes ids
nodes = {}
grounded_nodes = set()
for n in data["node_list"]:
    nid = n["node_id"]
    nodes[nid] = (n["point"]["X"], n["point"]["Y"], n["point"]["Z"])
    if n["is_grounded"] == 1:
        grounded_nodes.add(nid)

# create dict of all elements (rods) with tuple including their end node ids
elements = {}
for e in data["element_list"]:
    elements[e["element_id"]] = tuple(e["end_node_ids"])

# create graph structure
def build_graph(active_rods):
    graph = defaultdict(set)
    active_nodes = set()

    for eid in active_rods:
        n1, n2 = elements[eid]
        graph[n1].add(n2)
        graph[n2].add(n1)
        active_nodes.add(n1)
        active_nodes.add(n2)

    return graph, active_nodes

# check that rods are not flying
def is_valid_state(active_rods):
    graph, active_nodes = build_graph(active_rods)
    visited = set()

    for start in active_nodes:
        if start in visited:
            continue

        q = deque([start])
        has_ground = False

        while q:
            node = q.popleft()
            if node in visited:
                continue

            visited.add(node)
            if node in grounded_nodes:
                has_ground = True

            for neighbour in graph[node]:
                if neighbour not in visited:
                    q.append(neighbour)

        if not has_ground:
            return False

    return True


# use height as heuristicc
def heuristic(rod_id):
    n1, n2 = elements[rod_id]
    return 0.5 * (nodes[n1][2] + nodes[n2][2])

# greedy backward search
def backward_search():
    initial_state = frozenset(elements.keys())

    open_list = []
    counter = 0
    visited = set()

    # initialize search
    # counter use first in first out if same priority
    # save current state and rod to try
    for rod_id in initial_state:
        # needs to be negative because heapq uses smallest!
        priority = (len(initial_state), -heuristic(rod_id))
        heapq.heappush(open_list, (priority, counter, initial_state, rod_id, []))
        counter += 1

    while open_list:
        priority, counter, state, rod_id, sequence = heapq.heappop(open_list)

        # remove rod
        new_state = frozenset(state - {rod_id})

        if new_state in visited:
            continue
        visited.add(new_state)

        if not is_valid_state(new_state):
            continue

        # if it is a feasible option add rod to remove sequence
        new_sequence = sequence + [rod_id]

        # check if there are remaining nodes
        if len(new_state) == 0:
            return new_sequence

        # add all rods that could be removed to open_list 
        for next_rod in new_state:
            priority = (len(new_state), -heuristic(next_rod))
            heapq.heappush(
                open_list,
                (priority, counter, new_state, next_rod, new_sequence)
            )
            counter += 1

    return None


# plotting done by ChatGPT
def export_assembly_video(
    nodes,
    elements,
    sequence,
    output_path="assembly.mp4",
    fps=2,
    assembly=True,
    show_future=True,
):
    pts = np.array(list(nodes.values()), dtype=float)
    bounds_min = pts.min(axis=0)
    bounds_max = pts.max(axis=0)
    diag = np.linalg.norm(bounds_max - bounds_min)
    if diag == 0:
        diag = 1.0

    tube_radius = diag * 0.01
    node_radius = diag * 0.0

    plotter = pv.Plotter(off_screen=True, window_size=(1280, 720))
    plotter.open_movie(output_path, framerate=fps)
    plotter.set_background("white")

    if assembly:
        active_rods = set()
    else:
        active_rods = set(elements.keys())

    def draw_frame(step_idx, highlight_rod=None):
        plotter.clear()

        for eid, (n1, n2) in elements.items():
            p1 = nodes[n1]
            p2 = nodes[n2]
            tube = pv.Line(p1, p2).tube(radius=tube_radius)

            if eid == highlight_rod:
                plotter.add_mesh(tube, color="red")
            elif eid in active_rods:
                plotter.add_mesh(tube, color="blue")
            elif show_future:
                plotter.add_mesh(tube, color="lightgray", opacity=0.15)

        cloud = pv.PolyData(pts)
        node_mesh = cloud.glyph(scale=False, orient=False, geom=pv.Sphere(radius=node_radius))
        plotter.add_mesh(node_mesh, color="black")

        label = f"Step {step_idx}"
        if highlight_rod is not None:
            label += f" | Rod {highlight_rod}"
        #plotter.add_text(label, position="upper_left", font_size=14, color="black")

        #plotter.show_grid()
        plotter.reset_camera()
        plotter.write_frame()

    draw_frame(0)

    for i, rod in enumerate(sequence, start=1):
        print(i)
        if assembly:
            active_rods.add(rod)
        else:
            active_rods.remove(rod)

        draw_frame(i, highlight_rod=rod)

    plotter.close()
    print("Video exported")


sequence = backward_search()
print("Removal sequence:", sequence)
print("Assembly sequence:", list(reversed(sequence)) if sequence else None)
export_assembly_video(nodes, elements, reversed(sequence))
