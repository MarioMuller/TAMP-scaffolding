import json

# imports all the required data from the JSON
class Truss:
    def __init__(self, nodes, elements, grounded_nodes):
        self.nodes = nodes
        self.elements = elements
        self.grounded_nodes = grounded_nodes

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            data = json.load(f)

        nodes = {}
        grounded_nodes = set()
        elements = {}

        for n in data["node_list"]:
            nid = n["node_id"]
            nodes[nid] = (n["point"]["X"], n["point"]["Y"], n["point"]["Z"])
            if n.get("is_grounded", 0) == 1:
                grounded_nodes.add(nid)

        for e in data["element_list"]:
            elements[e["element_id"]] = tuple(e["end_node_ids"])

        return cls(nodes, elements, grounded_nodes)