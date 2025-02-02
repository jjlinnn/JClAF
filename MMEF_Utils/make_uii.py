import pickle
import scipy.sparse as sp
import networkx as nx
import numpy as np

with open("trnMat_sports.pkl", "rb") as f:
    ui_graph = pickle.load(f)

print(f"UI graph shape: {ui_graph.shape}")

similarity_file = "item_item_edges_sports.txt"
item_similarities = []

with open(similarity_file, 'r') as f:
    for line in f:
        item1, item2, similarity = line.strip().split()
        item1, item2 = int(item1), int(item2)
        similarity = float(similarity)
        if similarity >= 0.89:  
            item_similarities.append((item1, item2, similarity))

print(f"Number of item similarity edges with similarity >= 0.89: {len(item_similarities)}")

G_ii = nx.Graph()

for item1, item2, similarity in item_similarities:
    G_ii.add_edge(item1, item2, weight=1)

print(f"Number of edges in II graph: {len(G_ii.edges())}")

def expand_ui_to_uii(ui_graph, G_ii):
    rows, cols = ui_graph.nonzero()  
    new_edges = []

    for row, col in zip(rows, cols):
        user = row  
        item = col  

        if item in G_ii:
            similar_items = list(G_ii.neighbors(item))  
            for similar_item in similar_items:
                new_edges.append((user, similar_item))

    return new_edges

uii_edges = expand_ui_to_uii(ui_graph, G_ii)
print(f"Number of UII edges: {len(uii_edges)}")

num_users = ui_graph.shape[0]  
num_items = ui_graph.shape[1]  

rows = [edge[0] for edge in uii_edges]
cols = [edge[1] for edge in uii_edges]
data = [1.0] * len(uii_edges)  

num_nodes = num_users + num_items  
uii_sparse_matrix = sp.coo_matrix((data, (rows, cols)), shape=(num_users, num_items), dtype=np.float32)

print(f"UII graph shape: {uii_sparse_matrix.shape}")

with open("uii_graph_sports.pkl", "wb") as f:
    pickle.dump(uii_sparse_matrix, f)

print(f"UII graph saved as 'uii_graph_sports.pkl'.")
