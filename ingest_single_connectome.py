#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:50:24 2024

@author: alex
"""
import os
import torch
from torch_geometric.data import Data
import pandas as pd  # Import pandas
import pandas as pd
import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import networkx as nx
import numpy as np
from community import community_louvain  # Install using `pip install python-louvain`
import torch
from torch_geometric.data import Data
from torch_geometric.nn.models import Node2Vec



# Define the base directory and file name
base_dir=os.path.join('/mnt/newStor/paros/paros_WORK/GAT_alex/ADRC/data/ADRC_connectome_bank/connectome/DTI/plain/')

# Initialize lists for graphs and features
graphs = []
all_features = []

#file_name = 'ADRC0001_conn_plain.csv'

# Construct the file path robustly
file_path = os.path.join(base_dir, file_name)

# Debugging: Print the file path
print(f"Loading file from: {file_path}")

# Try to read the connectome matrix
try:
    connectome_matrix = pd.read_csv(file_path, header=None)
    print("Connectome matrix loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}.")
except pd.errors.EmptyDataError:
    print(f"Error: File at {file_path} is empty or cannot be parsed.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Ensure it's a square matrix
assert connectome_matrix.shape[0] == connectome_matrix.shape[1], "Matrix is not square!"

# Convert the connectome into an edge list
edge_indices = connectome_matrix.stack().reset_index()
edge_indices.columns = ['source', 'target', 'weight']

# Keep only non-zero edges (assuming no self-loops if the diagonal is zero)
edge_list = edge_indices[edge_indices['weight'] > 0]

# Convert to NetworkX graph for feature computation
G = nx.Graph()
for _, row in edge_list.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['weight'])

# Compute features for each node
node_features = {
    "degree": dict(G.degree(weight=None)),
    "clustering_coefficient": nx.clustering(G),
    "eigenvector_centrality": nx.eigenvector_centrality_numpy(G),
    "betweenness_centrality": nx.betweenness_centrality(G, normalized=True),
    "pagerank": nx.pagerank(G),
    "k_core": nx.core_number(G),
}

# Normalize features (optional)
import numpy as np

def normalize_features(features):
    arr = np.array(list(features.values()), dtype=np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min())

# Normalizing each feature
normalized_features = {key: normalize_features(values) for key, values in node_features.items()}

# Combine features into a feature matrix
num_nodes = connectome_matrix.shape[0]
feature_matrix = np.zeros((num_nodes, len(node_features)))

for i, (key, values) in enumerate(normalized_features.items()):
    for node in range(num_nodes):  # Iterate over node indices
        feature_matrix[node, i] = values[node]  # Access feature values using index




# Convert to PyTorch tensor
feature_matrix = torch.tensor(feature_matrix, dtype=torch.float)
# Convert to PyTorch tensor
feature_matrix = torch.tensor(feature_matrix, dtype=torch.float)

# Convert to PyTorch Geometric format
edge_index = torch.tensor(edge_list[['source', 'target']].values.T, dtype=torch.long)
edge_attr = torch.tensor(edge_list['weight'].values, dtype=torch.float)

# Build the graph data
graph_data = Data(x=feature_matrix, edge_index=edge_index, edge_attr=edge_attr)

# Print the graph object
print(graph_data)

# Compute Normalized Degree
degrees = np.array([deg for _, deg in G.degree()])
max_degree = degrees.max()
normalized_degrees = degrees / max_degree

# Compute Distance to Graph Center
center_nodes = nx.center(G)  # List of center nodes
shortest_path_lengths = [
    nx.shortest_path_length(G, source=center_nodes[0], target=node) for node in G.nodes()
]
mean_distance_to_center = np.mean(shortest_path_lengths)

# Louvain Clustering for Role Assignment
louvain_partition = community_louvain.best_partition(G)  # Dict: node -> community
unique_roles = len(set(louvain_partition.values()))  # Number of unique roles
role_distribution = np.array(list(louvain_partition.values()))

# Aggregate graph-level features
graph_features = {
    "mean_normalized_degree": np.mean(normalized_degrees),
    "mean_distance_to_center": mean_distance_to_center,
    "num_unique_roles": unique_roles,
    "role_distribution_entropy": -np.sum(
        np.bincount(role_distribution) / len(role_distribution)
        * np.log2(np.bincount(role_distribution) / len(role_distribution) + 1e-10)
    ),  # Entropy of role distribution
}

# Normalize features (optional)
graph_features = {key: np.float32(value) for key, value in graph_features.items()}

# Convert graph-level features to tensor
graph_feature_tensor = torch.tensor(list(graph_features.values()), dtype=torch.float)

# Add to PyTorch Geometric Data object
graph_data.graph_features = graph_feature_tensor

print("Node feature matrix shape:", graph_data.x.shape)
print("Graph feature tensor shape:", graph_data.graph_features.shape)
print("Edge index shape:", graph_data.edge_index.shape)
print("Edge attribute shape:", graph_data.edge_attr.shape)
print("Graph features:", graph_data.graph_features)
print("Node feature matrix (first 5 nodes):", graph_data.x[:5])


'''
#adding node embeddings


# Define Node2Vec model
embedding_dim = 128
node2vec = Node2Vec(
    edge_index=edge_index,  # Ensure zero-based and consecutive indices
    embedding_dim=embedding_dim,
    walk_length=10,
    context_size=5,
    walks_per_node=20,
    num_negative_samples=1,
    sparse=False,  # Use sparse=False for GPU support
)

# Train Node2Vec
loader = node2vec.loader(batch_size=64, shuffle=True)
optimizer = torch.optim.Adam(node2vec.parameters(), lr=0.01)

def train():
    node2vec.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = node2vec.loss(pos_rw, neg_rw)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Wrap training in a try-except block for debugging
try:
    for epoch in range(10):
        loss = train()
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')
except Exception as e:
    print(f"Error during Node2Vec training: {e}")

# Extract embeddings
try:
    node_embeddings = node2vec()
    print(f"Node embeddings shape: {node_embeddings.shape}")
except Exception as e:
    print(f"Error extracting embeddings: {e}")

# Concatenate embeddings with node features
try:
    combined_features = torch.cat((graph_data.x, node_embeddings), dim=1)
    graph_data.x = combined_features
    print("Node features updated with embeddings.")
except Exception as e:
    print(f"Error concatenating embeddings with features: {e}")
  #end adding node embeddings
    
 ''' 