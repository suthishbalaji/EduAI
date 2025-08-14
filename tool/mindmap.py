import matplotlib.pyplot as plt
import networkx as nx
import math

def radial_pos(G, root=None):
    """
    Compute radial positions for a tree-like mind map
    """
    if root is None:
        root = list(G.nodes)[0]

    def _radial_pos(node, radius=0, angle_start=0, angle_end=2*math.pi, pos=None, parent=None):
        if pos is None:
            pos = {}
        pos[node] = (radius * math.cos((angle_start+angle_end)/2), radius * math.sin((angle_start+angle_end)/2))
        children = list(G.successors(node))
        if children:
            n = len(children)
            for i, child in enumerate(children):
                angle0 = angle_start + i*(angle_end-angle_start)/n
                angle1 = angle_start + (i+1)*(angle_end-angle_start)/n
                pos = _radial_pos(child, radius+1.5, angle0, angle1, pos, node)
        return pos

    return _radial_pos(root)

def draw_radial_mindmap(graph):
    G = nx.DiGraph()
    
    # Add nodes
    for node in graph["nodes"]:
        G.add_node(node["id"], label=node["label"], color=node.get("color","#1f77b4"), emoji=node.get("emoji",""))
    
    # Add edges
    for edge in graph["edges"]:
        G.add_edge(edge["from"], edge["to"])
    
    # Compute radial positions
    pos = radial_pos(G, graph["root"])
    
    plt.figure(figsize=(10,10))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color="#555555")
    
    # Draw nodes
    for node in G.nodes(data=True):
        n_id = node[0]
        attrs = node[1]
        nx.draw_networkx_nodes(G, pos, nodelist=[n_id], node_size=2500, node_color=attrs["color"], alpha=0.9)
    
    # Draw labels with emojis
    labels = {n[0]: f"{n[1].get('emoji','')} {n[1]['label']}" for n in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')
    
    plt.axis('off')
    plt.title("VisionCraft Mind Map", fontsize=18)
    plt.show()


# ---- Example Mind Map JSON ----
mind_example = {
    "type":"mindmap",
    "root":"R1",
    "nodes":[
        {"id":"R1","label":"VisionCraft","emoji":"🤖","color":"#1f77b4"},
        {"id":"N1","label":"Hardware","emoji":"💻","color":"#ff7f0e"},
        {"id":"N2","label":"Software","emoji":"🖥️","color":"#2ca02c"},
        {"id":"N3","label":"Sensors","emoji":"📡","color":"#d62728"},
        {"id":"N4","label":"Camera","emoji":"📷","color":"#9467bd"},
        {"id":"N5","label":"AI Features","emoji":"🧠","color":"#8c564b"},
        {"id":"N6","label":"Battery","emoji":"🔋","color":"#e377c2"},
        {"id":"N7","label":"User Interface","emoji":"🎛️","color":"#17becf"}
    ],
    "edges":[
        {"from":"R1","to":"N1"},
        {"from":"R1","to":"N2"},
        {"from":"R1","to":"N7"},
        {"from":"N1","to":"N3"},
        {"from":"N1","to":"N6"},
        {"from":"N3","to":"N4"},
        {"from":"N2","to":"N5"}
    ]
}

# Draw the mind map
draw_radial_mindmap(mind_example)
