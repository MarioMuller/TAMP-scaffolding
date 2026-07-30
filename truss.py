import json

# imports the truss structure based on the JSON
# TODO: replace this  with the actual building elements

class Truss:
    def __init__(self, nodes, elements, grounded_rods, couplers):
        self.nodes = nodes
        self.elements = elements
        self.grounded_rods = grounded_rods
        self.couplers = couplers

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            data = json.load(f)

        nodes = {}
        rods = {}
        grounded_rods = set()
        couplers = set()
        
        for n in data["node_list"]:
            nid = n["node_id"]
            nodes[nid] = (n["point"]["X"], n["point"]["Y"], n["point"]["Z"])

        for e in data["rod_list"]:
            rods[e["rod_id"]] = tuple(e["end_node_ids"])
            if e.get("grounded", 0) == 1:
                grounded_rods.add(e["rod_id"])
                
        for c in data.get("coupler_list", []):
            rod_1, rod_2 = c["rod_ids"]

            # Store every pair in a consistent order.
            couplers.add(tuple(sorted((rod_1, rod_2))))      
            
        # print(f"couplers: {couplers}")  

        return cls(nodes, rods, grounded_rods, couplers)
