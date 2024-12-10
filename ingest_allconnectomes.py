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



import os
import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import networkx as nx
from community import community_louvain

import os
import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import networkx as nx
from community import community_louvain

# Base directory containing connectome files
base_dir = '/mnt/newStor/paros/paros_WORK/GAT_alex/ADRC/data/ADRC_connectome_bank/connectome/DTI/plain/'

# Initialize lists for graphs and features
graphs = []
all_features = []
all_graph_features = []

# Count .csv files
subject_files = [f for f in os.listdir(base_dir) if f.startswith('ADRC') and f.endswith('.csv')]
num_subjects = len(subject_files)

print(f"Number of subjects in the dataset: {num_subjects}")


# Process all files in the directory
#for file_name in os.listdir(base_dir):
for file_name in subject_files:
    if file_name.endswith('.csv'):  # Ensure the file has a .csv extension
        file_path = os.path.join(base_dir, file_name)
        print(f"Processing file: {file_path}")
        #subject_id = file_name.split('_')[-1].replace('.pt', '')  # Customize based on file naming convention
        subject_id = file_name.split('_')[0]
        # Load connectome matrix
        try:
            connectome_matrix = pd.read_csv(file_path, header=None).values
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Ensure the matrix is square
        if connectome_matrix.shape[0] != connectome_matrix.shape[1]:
            print(f"Skipping {file_path}: Matrix is not square.")
            continue

        # Create edge list
        edge_indices = np.stack(np.nonzero(connectome_matrix), axis=1)
        edge_weights = connectome_matrix[edge_indices[:, 0], edge_indices[:, 1]]

        # Build NetworkX graph
        G = nx.from_numpy_matrix(connectome_matrix)

        # Compute node features
        node_features = {
            "degree": dict(G.degree(weight=None)),
            "clustering_coefficient": nx.clustering(G),
            "eigenvector_centrality": nx.eigenvector_centrality_numpy(G),
            "betweenness_centrality": nx.betweenness_centrality(G, normalized=True),
            "pagerank": nx.pagerank(G),
            "k_core": nx.core_number(G),
        }

        # Combine node features into a feature matrix
        num_nodes = connectome_matrix.shape[0]
        feature_matrix = np.zeros((num_nodes, len(node_features)))

        for i, (key, values) in enumerate(node_features.items()):
            for node in range(num_nodes):
                feature_matrix[node, i] = values[node]

        # Append feature matrix for global normalization
        all_features.append(feature_matrix)

        # Compute graph-level features
        degrees = np.array([deg for _, deg in G.degree()])
        max_degree = degrees.max()
        normalized_degrees = degrees / max_degree

        center_nodes = nx.center(G)  # List of center nodes
        shortest_path_lengths = [
            nx.shortest_path_length(G, source=center_nodes[0], target=node) for node in G.nodes()
        ]
        mean_distance_to_center = np.mean(shortest_path_lengths)

        louvain_partition = community_louvain.best_partition(G)  # Dict: node -> community
        unique_roles = len(set(louvain_partition.values()))  # Number of unique roles
        role_distribution = np.array(list(louvain_partition.values()))

        graph_features = {
            "mean_normalized_degree": np.mean(normalized_degrees),
            "mean_distance_to_center": mean_distance_to_center,
            "num_unique_roles": unique_roles,
            "role_distribution_entropy": -np.sum(
                np.bincount(role_distribution) / len(role_distribution)
                * np.log2(np.bincount(role_distribution) / len(role_distribution) + 1e-10)
            ),
        }

        graph_features_tensor = torch.tensor(list(graph_features.values()), dtype=torch.float)
        all_graph_features.append(graph_features_tensor)

        # Convert to PyTorch tensors
        edge_index = torch.tensor(edge_indices.T, dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        feature_matrix = torch.tensor(feature_matrix, dtype=torch.float)

        # Create PyTorch Geometric Data object
        graph_data = Data(x=feature_matrix, edge_index=edge_index, edge_attr=edge_attr)
        graph_data.graph_features = graph_features_tensor  # Add graph-level features
        graph_data.id = subject_id  # Add as an attribute
        print(f"Assigned Subject ID {subject_id} to graph from file {file_name}")

        graphs.append(graph_data)

# Stack all feature matrices for global normalization
all_features = np.vstack(all_features)

# Compute global min and max
global_min = all_features.min(axis=0)
global_max = all_features.max(axis=0)

# Normalize features across all graphs
for graph in graphs:
    graph.x = (graph.x - torch.tensor(global_min)) / (torch.tensor(global_max) - torch.tensor(global_min))

# Print the first graph to verify
print("First graph:")
print(graphs[0])
print("Normalized node features (first 5 nodes):")
print(graphs[0].x[:5])
print("Graph features (first graph):")
print(graphs[0].graph_features)


# File path to save the graphs
output_path = "/mnt/newStor/paros/paros_WORK/GAT_alex/ADRC/data/data4GNN/ADRC_all_subj_graph_data.pt"

# Save the list of graphs
torch.save(graphs, output_path)

print(f"Graph data saved to {output_path}")
