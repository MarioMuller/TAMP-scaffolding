import json

# imports the truss structure based on the JSON
# TODO: replace this  with the actual building elements

class Truss:
    def __init__(self, nodes, elements, grounded_nodes, node_fixities=None):
        self.nodes = nodes
        self.elements = elements
        self.grounded_nodes = grounded_nodes
        self.node_fixities = node_fixities or {}

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            data = json.load(f)

        nodes = {}
        rods = {}

        for n in data["node_list"]:
            nid = n["node_id"]
            nodes[nid] = (n["point"]["X"], n["point"]["Y"], n["point"]["Z"])


        for e in data["rod_list"]:
            rods[e["rod_id"]] = tuple(e["end_node_ids"])
            if n.get("is_grounded", 0) == 1:
                grounded_rods.add(nid)

        return cls(nodes, rods, grounded_nodes, node_fixities=node_fixities)
