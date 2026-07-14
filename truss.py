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
        grounded_nodes = set()
        node_fixities = {}
        elements = {}

        for n in data["node_list"]:
            nid = n["node_id"]
            nodes[nid] = (n["point"]["X"], n["point"]["Y"], n["point"]["Z"])
            node_fixities[nid] = tuple(n.get("fixities", []))
            if n.get("is_grounded", 0) == 1:
                grounded_nodes.add(nid)

        for e in data["element_list"]:
            elements[e["element_id"]] = tuple(e["end_node_ids"])

        return cls(nodes, elements, grounded_nodes, node_fixities=node_fixities)
