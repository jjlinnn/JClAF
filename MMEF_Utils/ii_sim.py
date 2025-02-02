import torch
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

with open('multimodal_embeddings_sports.pkl', 'rb') as f:
    embeddings = pickle.load(f)

item_ids = list(embeddings.keys())
#embedding_vectors = [embedding.squeeze().numpy() for embedding in embeddings.values()]  
embedding_vectors = [embedding.squeeze().cpu().numpy() for embedding in embeddings.values()]  

embedding_matrix = np.vstack(embedding_vectors)

similarity_matrix = cosine_similarity(embedding_matrix)

similarity_threshold = 0.84  

edges = []
for i in range(len(item_ids)):
    for j in range(i + 1, len(item_ids)):  
        if similarity_matrix[i, j] > similarity_threshold:
            edges.append((item_ids[i], item_ids[j], similarity_matrix[i, j])) 

G = nx.Graph()
G.add_weighted_edges_from(edges)

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

with open('item_item_edges_sports.txt', 'w') as f:
    for edge in edges:
        f.write(f"{edge[0]} {edge[1]} {edge[2]:.4f}\n")

nx.write_gpickle(G, "item_item_graph_sports.gpickle")
