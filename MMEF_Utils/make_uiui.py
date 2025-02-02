import pandas as pd
import networkx as nx
import scipy.sparse as sp
import pickle
from collections import defaultdict
import numpy as np

def build_expanded_ui_graph(df):
    train_df = df[df['x_label'] == 0]

    G_ui = nx.Graph()
    G_ui.add_edges_from(train_df[['userID', 'itemID']].values)
    
    user_to_items = defaultdict(set)
    item_to_users = defaultdict(set)
    for user, item in train_df[['userID', 'itemID']].values:
        user_to_items[user].add(item)
        item_to_users[item].add(user)

    new_edges = set()
    for user, items in user_to_items.items():
        for item in items:
            for other_user in item_to_users[item]:
                if other_user != user:
                    new_items = user_to_items[other_user]  
                    for new_item in new_items:
                        new_edges.add((user, new_item)) 

    G_ui.add_edges_from(new_edges)
    
    return G_ui

def save_graph_to_pkl(G, filename):
    edges = list(G.edges())
    rows = [edge[1] for edge in edges] 
    cols = [edge[0] for edge in edges] 
    data = [1.0] * len(edges) 

    user_count = max(rows) + 1 
    item_count = max(cols) + 1  
    sparse_matrix = sp.coo_matrix((data, (rows, cols)), shape=(user_count,item_count), dtype=np.float32)

    with open(filename, "wb") as f:
        pickle.dump(sparse_matrix, f)
    print(f"Sparse matrix saved as '{filename}' with itemID as rows and userID as columns.")

df = pd.read_csv("sports-indexed-v4.inter", sep="\t")

expanded_graph = build_expanded_ui_graph(df)

save_graph_to_pkl(expanded_graph, "expanded_uiui_graph_sports.pkl")
