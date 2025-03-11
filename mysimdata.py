import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Parameters
num_subjects = 80
num_nodes = 100  # Brain regions
num_edges = 300  # Approximate number of edges per graph
num_eigenes = 10  # Gene expression summary features (eigengenes)
mean_age = 55  # Mean age for Gaussian distribution
std_age = 12  # Standard deviation for age

# Directory to save generated graphs
output_dir = "simulated_brain_graphs"
os.makedirs(output_dir, exist_ok=True)

# Generate simulated data
all_graphs = []
all_metadata = []

for i in range(num_subjects):
    # Assign sex (half female, half male)
    sex = "Female" if i < num_subjects // 2 else "Male"
    
    # Assign APOE4 status (half carriers, half non-carriers)
    apoe4_status = "Carrier" if i % 2 == 0 else "Non-Carrier"
    
    # Generate age from a Gaussian distribution
    age = int(np.clip(np.random.normal(mean_age, std_age), 25, 85))
    
    eigengenes = np.random.rand(num_eigenes)  # Random eigengene expression
    
    # Generate a random graph structure
    G = nx.erdos_renyi_graph(num_nodes, p=0.1)  # Random connectivity
    
    # Adjust connectivity for APOE4 females (decrease clustering coefficient in 25% of nodes)
    if sex == "Female" and apoe4_status == "Carrier":
        affected_nodes = np.random.choice(num_nodes, size=int(num_nodes * 0.25), replace=False)
        for node in affected_nodes:
            neighbors = list(G.neighbors(node))
            if len(neighbors) > 1:
                G.remove_edges_from([(node, n) for n in neighbors[:len(neighbors)//2]])
    
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    
    # Node features: Random structural features + eigengenes appended to each node
    node_features = np.random.rand(num_nodes, 5)  # Example 5 SC features per node
    eigengene_features = np.tile(eigengenes, (num_nodes, 1))  # Assign same eigengenes to all nodes
    x = torch.tensor(np.hstack([node_features, eigengene_features]), dtype=torch.float)
    
    # Create graph data object
    graph_data = Data(x=x, edge_index=edge_index)
    
    # Save data
    file_path = os.path.join(output_dir, f"subject_{i}.pt")
    torch.save(graph_data, file_path)
    
    all_graphs.append(graph_data)
    all_metadata.append((i, age, sex, apoe4_status, eigengenes.tolist()))

# Save metadata
metadata_path = os.path.join(output_dir, "metadata.csv")
np.savetxt(metadata_path, all_metadata, delimiter=",", fmt="%s", header="subject_id,age,sex,apoe4_status," + ",".join([f"eigengene_{i}" for i in range(num_eigenes)]), comments="")

print(f"Generated {num_subjects} brain graphs with eigengenes, sex, and APOE4 status, with APOE4 females showing decreased connectivity in 25% of nodes. Data saved in {output_dir}/")
