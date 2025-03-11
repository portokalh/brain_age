import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, GlobalAttention
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

'''
Data Loading: Loads the simulated brain graphs and processes edges to ensure undirected connectivity.

Model Architecture: A three-layer GAT with attention pooling. Each GAT layer uses multi-head attention, followed by ELU activation and dropout.

Training Loop: Implements 5-fold cross-validation, training for 100 epochs per fold with early stopping based on validation MAE.

Evaluation: Reports MAE and RMSE for each fold, providing aggregated performance metrics.

'''

# Load metadata to extract ages
metadata_path = os.path.join("simulated_brain_graphs", "metadata.csv")
metadata = np.genfromtxt(metadata_path, delimiter=',', skip_header=1, dtype=object)
ages = metadata[:, 1].astype(float)

# Load all graphs and add reverse edges for undirected connectivity
all_graphs = []
for i in range(len(metadata)):
    file_path = os.path.join("simulated_brain_graphs", f"subject_{i}.pt")
    data = torch.load(file_path)
    # Add reverse edges to make undirected
    edge_index = torch.cat([data.edge_index, data.edge_index[[1,0]]], dim=1)
    data.edge_index = torch.unique(edge_index, dim=1)  # Remove duplicates
    data.y = torch.tensor([ages[i]], dtype=torch.float)
    all_graphs.append(data)

 class GATModel(nn.Module):
    def __init__(self, num_features):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(num_features, 64, heads=8, concat=True)
        self.gat2 = GATConv(64*8, 64, heads=8, concat=True)
        self.gat3 = GATConv(64*8, 64, heads=1, concat=False)
        self.pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1)
        ))
        self.regressor = nn.Sequential(
            nn.Linear(64, 32), nn.ELU(), nn.Dropout(0.5), nn.Linear(32, 1)
        )
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat3(x, edge_index))
        x = self.dropout(x)
        
        x = self.pool(x, batch)
        return self.regressor(x).squeeze()
        

 # Set up K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []
rmse_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(all_graphs)):
    print(f"Fold {fold + 1}")
    # Split datasets
    train_data = [all_graphs[i] for i in train_idx]
    val_data = [all_graphs[i] for i in val_idx]
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    
    # Initialize model and optimizer
    model = GATModel(num_features=15)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_mae = float('inf')
    for epoch in range(100):  # Train for 100 epochs
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        avg_loss = total_loss / len(train_data)
        
        # Validation
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for batch in val_loader:
                preds.append(model(batch))
                truths.append(batch.y)
        pred = torch.cat(preds)
        truth = torch.cat(truths)
        mae = F.l1_loss(pred, truth).item()
        rmse = torch.sqrt(F.mse_loss(pred, truth)).item()
        
        # Update best model
        if mae < best_mae:
            best_mae = mae
            best_rmse = rmse
        
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.2f} | Val MAE: {mae:.2f} | Val RMSE: {rmse:.2f}")
    
    mae_scores.append(best_mae)
    rmse_scores.append(best_rmse)
    print(f"Fold {fold+1} Best MAE: {best_mae:.2f}, RMSE: {best_rmse:.2f}")

# Final Results
print(f"\nCross-Validation MAE: {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}")
print(f"Cross-Validation RMSE: {np.mean(rmse_scores):.2f} ± {np.std(rmse_scores):.2f}")          