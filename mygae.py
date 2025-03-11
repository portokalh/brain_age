import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import zscore

# ------------------------
# 1. Preprocessing Functions
# ------------------------

def threshold_connectome(adj_matrix, sparsity=0.1):
    """Thresholds a connectivity matrix, keeping the top X% strongest connections."""
    num_edges = int(sparsity * adj_matrix.size)  # Compute number of edges to keep
    sorted_edges = np.sort(adj_matrix.flatten())[::-1]  # Sort in descending order
    threshold_value = sorted_edges[num_edges]  # Find the cutoff threshold
    return np.where(adj_matrix >= threshold_value, adj_matrix, 0)  # Apply threshold

def normalize_connectome_zscore(adj_matrix):
    """Z-score normalization across rows (removes inter-subject variability)."""
    return zscore(adj_matrix, axis=1, ddof=1)

def preprocess_connectome(adj_matrix, sparsity=0.1, normalization="zscore"):
    """Apply thresholding and normalization to a connectome."""
    adj_matrix = threshold_connectome(adj_matrix, sparsity=sparsity)  # Step 1: Threshold
    if normalization == "zscore":
        adj_matrix = normalize_connectome_zscore(adj_matrix)
    return (adj_matrix + adj_matrix.T) / 2  # Ensure symmetry

# ------------------------
# 2. Load and Prepare Connectome Data
# ------------------------
print("Loading and Preprocessing Data...")

# Simulated example: Replace with real dataset
num_subjects, num_regions = 100, 90  # 100 subjects, 90 brain regions
connectomes = np.random.rand(num_subjects, num_regions, num_regions)  # Simulated connectomes
ages = np.random.randint(20, 80, num_subjects)  # Simulated brain ages

# Apply preprocessing
connectomes = np.array([preprocess_connectome(c) for c in connectomes])

# Convert connectomes into PyTorch Geometric graphs
graph_data_list = []
for i in range(num_subjects):
    adj_matrix = connectomes[i]
    sparse_matrix = sp.coo_matrix(adj_matrix)
    edge_index, edge_attr = from_scipy_sparse_matrix(sparse_matrix)

    graph_data = Data(
        x=torch.tensor(adj_matrix, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(ages[i], dtype=torch.float),
    )
    graph_data_list.append(graph_data)

print(f"Loaded {len(graph_data_list)} preprocessed connectomes.")

# ------------------------
# 3. Define Graph Autoencoder (GAE)
# ------------------------
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x  # Latent representation

# Create the GAE model
hidden_dim = 32
model = GAE(GCNEncoder(in_channels=num_regions, hidden_dim=hidden_dim))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

# ------------------------
# 4. Train GAE with Cross-Validation
# ------------------------
def train(graph_data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(graph_data.x, graph_data.edge_index)
    loss = model.recon_loss(z, graph_data.edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()

num_epochs = 50
kf = KFold(n_splits=5, shuffle=True, random_state=42)

all_mae_scores = []
for fold, (train_idx, test_idx) in enumerate(kf.split(graph_data_list)):
    print(f"Fold {fold + 1}: Training GAE...")
    
    # Split data
    train_data = [graph_data_list[i] for i in train_idx]
    test_data = [graph_data_list[i] for i in test_idx]

    # Train model
    for epoch in range(num_epochs):
        total_loss = np.mean([train(graph) for graph in train_data])
    print(f"Fold {fold + 1}, Final GAE Loss: {total_loss:.4f}")

    # ------------------------
    # 5. Extract Embeddings and Predict Age
    # ------------------------
    print(f"Fold {fold + 1}: Extracting Embeddings...")
    model.eval()
    X_train = np.array([model.encode(graph.x, graph.edge_index).detach().cpu().numpy().mean(axis=0) for graph in train_data])
    y_train = np.array([graph.y.item() for graph in train_data])
    
    X_test = np.array([model.encode(graph.x, graph.edge_index).detach().cpu().numpy().mean(axis=0) for graph in test_data])
    y_test = np.array([graph.y.item() for graph in test_data])

    print(f"Fold {fold + 1}: Training Random Forest...")
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    all_mae_scores.append(mae)
    print(f"Fold {fold + 1}, MAE: {mae:.2f} years")

print(f"Final Mean Absolute Error (5-fold CV): {np.mean(all_mae_scores):.2f} years")

# ------------------------
# 6. Extract Important Subgraph
# ------------------------
def get_important_edges(threshold=0.05):
    model.eval()
    with torch.no_grad():
        z = model.encode(graph_data_list[0].x, graph_data_list[0].edge_index)
        edge_probs = model.decoder(z, graph_data_list[0].edge_index)

    edge_importance = torch.abs(edge_probs - graph_data_list[0].edge_attr).cpu().numpy()
    important_edges = graph_data_list[0].edge_index[:, edge_importance > threshold]
    return important_edges

important_edges = get_important_edges()

# Convert to NetworkX for visualization
G = nx.Graph()
G.add_edges_from(important_edges.T.numpy())

plt.figure(figsize=(8, 6))
nx.draw(G, node_size=50, edge_color='blue', alpha=0.6)
plt.title("Most Predictive Subgraph for Brain Age")
plt.show()



'''
# ------------------------
# 6. Brain Atlas Visualization
# ------------------------
atlas = datasets.fetch_atlas_destrieux_2009()
region_coords = atlas.region_coords

def plot_brain_graph(important_edges, node_coords, node_size=50):
    G = nx.Graph()
    edges = important_edges.cpu().numpy().T
    for i, coord in enumerate(node_coords):
        G.add_node(i, pos=(coord[0], coord[1]))
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    pos = {i: (coord[0], coord[1]) for i, coord in enumerate(node_coords)}
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, node_size=node_size, edge_color='blue', alpha=0.6, width=1.5)
    plt.title("Most Predictive Subgraph for Brain Age")
    plt.show()

plot_brain_graph(important_edges, region_coords)
'''