from json_import import Truss
from collections import defaultdict, deque
import pyvista as pv
import numpy as np
import heapq

class TrussSearch:
    def __init__(self, truss):
        self.truss = truss

    # create graph structure
    def build_graph(self, active_rods):
        graph = defaultdict(set)
        active_nodes = set()

        for eid in active_rods:
            n1, n2 = self.truss.elements[eid]
            graph[n1].add(n2)
            graph[n2].add(n1)
            active_nodes.add(n1)
            active_nodes.add(n2)

        return graph, active_nodes

    # check that rods are not flying
    def is_valid_state(self, active_rods):
        graph, active_nodes = self.build_graph(active_rods)
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
                if node in self.truss.grounded_nodes:
                    has_ground = True

                for neighbour in graph[node]:
                    if neighbour not in visited:
                        q.append(neighbour)

            if not has_ground:
                return False

        return True


    # use height as heuristicc
    def heuristic(self, rod_id):
        n1, n2 = self.truss.elements[rod_id]
        return 0.5 * (self.truss.nodes[n1][2] + self.truss.nodes[n2][2])

    # greedy backward search
    def backward_search(self):
        initial_state = frozenset(self.truss.elements.keys())

        open_list = []
        counter = 0
        visited = set()

        # initialize search
        # counter use first in first out if same priority
        # save current state and rod to try
        for rod_id in initial_state:
            # needs to be negative because heapq uses smallest!
            priority = (len(initial_state), -self.heuristic(rod_id))
            heapq.heappush(open_list, (priority, counter, initial_state, rod_id, []))
            counter += 1

        while open_list:
            priority, counter, state, rod_id, sequence = heapq.heappop(open_list)

            # remove rod
            new_state = frozenset(state - {rod_id})

            if new_state in visited:
                continue
            visited.add(new_state)

            if not self.is_valid_state(new_state):
                continue

            # if it is a feasible option add rod to remove sequence
            new_sequence = sequence + [rod_id]

            # check if there are remaining nodes
            if len(new_state) == 0:
                return new_sequence

            # add all rods that could be removed to open_list 
            for next_rod in new_state:
                priority = (len(new_state), -self.heuristic(next_rod))
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
    sequence = list(sequence)
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


if __name__ == "__main__":
    truss = Truss.from_json("JSON/long_beam_test.json")

    searcher = TrussSearch(truss)

    removal_sequence = searcher.backward_search()
    assembly_sequence = list(reversed(removal_sequence)) if removal_sequence else None

    print("Removal:", removal_sequence)
    print("Assembly:", assembly_sequence)

    if assembly_sequence:
        export_assembly_video(truss.nodes, truss.elements, assembly_sequence)