import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d

#from torch_geometric.nn import TopKPooling
#from topk_pool_hsm import TopKPooling

from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, TransformerConv, GATConv
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_sparse import spspmm
import numpy as np

#from net.braingraphconv import MyNNConv

#TopKPooling2 = TopKPooling

##########################################################################################################################



#not used
class GATNet_orig(torch.nn.Module):
    def __init__(self, indim, ratio, nclass, batchSize, k=6, R=332):
        super(GATNet, self).__init__()

        out_channels = 64  # Set the number of output channels for GAT layers
        num_heads_1 = 32  # Change number of heads for the first GATConv layer

        # GATConv layers
        # Note that the output dimension after gat_conv1 is out_channels * num_heads_1
        self.gat_conv1 = GATConv(in_channels=84, out_channels=out_channels // num_heads_1, heads=num_heads_1, concat=True)

        # For gat_conv2, we need to account for the output dimension from gat_conv1
        self.gat_conv2 = GATConv(out_channels, out_channels=out_channels, heads=1, concat=True)

        # Fully connected layer to output a single number after pooling
        self.fc = torch.nn.Linear(out_channels, 1)  # Single output node for regression


    def forward(self, x, edge_index, batch):
        # First GAT layer
        x = self.gat_conv1(x, edge_index)
        x = F.elu(x)  # Apply non-linearity

        # Second GAT layer
        x = self.gat_conv2(x, edge_index)

        # Global mean pooling to get graph-level features
        x = gap(x, batch)

        # Final fully connected layer to produce a single prediction per graph
        x = self.fc(x)

        return x, torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)  # Return a vector of shape [batch_size]







#not used
class GAT_example(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GAT, self).__init__()
        # First GAT layer
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=8, dropout=0.2)
        # Second GAT layer
        self.conv2 = GATConv(hidden_channels * 8, num_classes, heads=1, concat=False, dropout=0.2)

    def forward(self, x, edge_index, batch):
        # Apply the first GAT layer
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        # Apply the second GAT layer
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x




###edit this alex
# Custom GAT layer that returns attention weights
class CustomGATConv(GATConv):
    def forward(self, x, edge_index, return_attention_weights=True):
        # Use the parent class forward method with the option to return attention weights
        x, (edge_index, att_weights) = super().forward(x, edge_index, return_attention_weights=return_attention_weights)
        return x, att_weights

class GATNet(torch.nn.Module):
    def __init__(self, indim, ratio, nclass, batchSize, k=6, R=332):
        super(GATNet, self).__init__()

        out_channels = 64  # Set the number of output channels for GAT layers
        num_heads_1 = 32  # Number of heads for the first GATConv layer

        out_channels = 16  # Set the number of output channels for GAT layers
        num_heads_1 = 8  # Number of heads for the first GATConv layer

        # Custom GATConv layers to get attention weights
        self.gat_conv1 = CustomGATConv(in_channels=84, out_channels=out_channels // num_heads_1, heads=num_heads_1, concat=True)
        self.gat_conv2 = CustomGATConv(out_channels, out_channels=out_channels, heads=1, concat=True)

        


        self.dropout = torch.nn.Dropout(p=0.2)  # 50% dropout rate, adjust as needed




        # Fully connected layer to output a single number after pooling
        self.fc = torch.nn.Linear(out_channels, 1)  # Single output node for regression

    def forward(self, x, edge_index, batch):
        # First GAT layer with edge attention scores
        x, edge_score_map1 = self.gat_conv1(x, edge_index)
        x = F.elu(x)  # Apply non-linearity

        x = self.dropout(x)

        # Second GAT layer with edge attention scores
        x, edge_score_map2 = self.gat_conv2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        # Average over the heads in edge_score_map1 to get a single score per edge
        edge_score_map_1 = edge_score_map1.mean(dim=1, keepdim=True)  # Ensures shape [28224, 1]
        edge_score_map_1 = edge_score_map_1.view(batch.max().item() + 1, 84, 84)


        edge_score_map_2 = edge_score_map2.view(batch.max().item() + 1, 84, 84)

        edge_score_map = (edge_score_map_1 + edge_score_map_2) / 2
        #print("Combined edge_score_map:", edge_score_map.size())

        # Global mean pooling for graph-level features
        x = gap(x, batch)

        # Final fully connected layer to produce a single prediction per graph
        x = self.fc(x)

        return x, edge_score_map_1, edge_score_map_2, edge_score_map







