import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, TransformerConv, GATConv
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_sparse import spspmm
import numpy as np
import os
import pandas as pd



##########################################################################################################################

###edit this alex
# Custom GAT layer with attention weights
class CustomGATConv(GATConv):
    def forward(self, x, edge_index, return_attention_weights=True):
        # Use parent class forward method with option to return attention weights
        x, (edge_index, att_weights) = super().forward(x, edge_index, return_attention_weights=return_attention_weights)
        return x, att_weights

class GATNet(torch.nn.Module):
    def __init__(self, indim, ratio, nclass, batchSize, temperature, lambda_entropy):
        super(GATNet, self).__init__()

        self.temperature = temperature  # Temperature for sharpening
        self.lambda_entropy = lambda_entropy  # Weight for entropy regularization


        ## 128 * 8 = 1024
        out_channels_1 =  64 # 16 # 64  # Set the number of output channels for GAT layers
        num_heads_1 = 16  # Number of heads for the first GATConv layer
        # 64 * 8 = 512
        out_channels_2 = 32 # 8 # 32  # Set the number of output channels for GAT layers
        num_heads_2 = 8  # Number of heads for the first GATConv layer
        # 32 * 4 = 128
        out_channels_3 = 16 #16  # Set the number of output channels for GAT layers
        num_heads_3 = 4 #2  # Number of heads for the first GATConv layer

        # Custom GATConv layers to get attention weights
        self.gat_conv1 = CustomGATConv(in_channels=84, out_channels=out_channels_1 // num_heads_1, heads=num_heads_1, concat=True)
        self.gat_conv2 = CustomGATConv(in_channels=out_channels_1, out_channels=out_channels_2 // num_heads_2, heads=num_heads_2, concat=True)
        self.gat_conv3 = CustomGATConv(in_channels=out_channels_2, out_channels=out_channels_3 // num_heads_3, heads=num_heads_3, concat=True)
       
        
           # Linear layer to match dimensions for skip connections
        self.align1 = nn.Linear(84, out_channels_1)  # Align input_x to 64
        self.align2 = nn.Linear(out_channels_1, out_channels_2)  # Align for second skip connection

        # Layer normalization for stability
        self.norm1 = torch.nn.LayerNorm(out_channels_1)  # After first GAT layer
        self.norm2 = torch.nn.LayerNorm(out_channels_2)  # After second GAT layer
        self.norm3 = torch.nn.LayerNorm(out_channels_3)  # After third GAT layer

        # Dropout layers
        self.dropout1 = torch.nn.Dropout(p=0.2)  #0.1
        self.dropout2 = torch.nn.Dropout(p=0.2) #0.05
        self.dropout3 = torch.nn.Dropout(p=0.2) #0.05

        # Fully connected layer to output a single number after pooling
        #self.fc = torch.nn.Linear(2*out_channels_3, 1)  # Single output node for regression
        self.fc = torch.nn.Linear(out_channels_3, 1)

    def compute_entropy(self, attention_scores):
        """
        Compute entropy of attention scores for regularization.
        """
        attention_scores = F.softmax(attention_scores, dim=-1)  # Ensure normalization
        entropy = -torch.sum(attention_scores * torch.log(attention_scores.clamp(min=1e-10)), dim=-1)
        return entropy.mean()

    def compute_loss(self, predictions, targets, attention_scores_list):
        """
        Compute loss with entropy regularization.
        """
        #task_loss = F.mse_loss(predictions, targets)
        task_loss = F.smooth_l1_loss(predictions, targets)
        entropy_loss = sum([self.compute_entropy(att_scores) for att_scores in attention_scores_list])
        total_loss = task_loss + self.lambda_entropy * entropy_loss
        return total_loss

    @staticmethod
    def normalize_scores(scores, scale=100):
        """
        Normalize scores to the range [0, scale].
        """
        min_score = scores.min()
        max_score = scores.max()
        normalized_scores = ((scores - min_score) / (max_score - min_score)) * scale
        return normalized_scores




    def forward(self, x, edge_index, batch):
        
        #skip connections
        input_x = x
        
        # First GAT layer with edge attention scores
        x, edge_score_map1 = self.gat_conv1(x, edge_index)
        x = F.elu(x)  # Apply non-linearity
        x = self.norm1(x)
        x = self.dropout1(x)
        x = x + self.align1(input_x)   # Skip connection (residual connection)
        

        # Second GAT layer with edge attention scores
        input_x2 = x
        x, edge_score_map2 = self.gat_conv2(x, edge_index)
        x = F.elu(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        x = x + self.align2(input_x2)  # Skip connection

         # Third GAT layer with edge attention scores
        x, edge_score_map3 = self.gat_conv3(x, edge_index)
        x = F.elu(x)
        x = self.norm3(x)
        x = self.dropout3(x)

        '''
        print(f"Shape after gat_conv1: {x.shape}")
        print(f"Shape after norm1: {x.shape}")
        print(f"Shape after gat_conv2: {x.shape}")
        print(f"Shape after norm2: {x.shape}")
        print(f"Shape after gat_conv3: {x.shape}")
        print(f"Shape after norm3: {x.shape}")
        print(f"batch: {batch}")
        print(f"edge_index: {edge_index.shape}")

        
      
        print(f"edge_score_map1 shape: {edge_score_map1.shape}")
        print(f"edge_score_map2 shape: {edge_score_map2.shape}")
        print(f"edge_score_map3 shape: {edge_score_map3.shape}")
        '''

        edge_score_map1 = edge_score_map1.mean(dim=1, keepdim=False)  
        edge_score_map2 = edge_score_map2.mean(dim=1, keepdim=False)  
        edge_score_map3 = edge_score_map3.mean(dim=1, keepdim=False)  

        edge_index_np = edge_index.detach().cpu().numpy()
        edge_score_map3_np = edge_score_map3.detach().cpu().numpy()
        num_edges = min(edge_index_np.shape[1], edge_score_map3_np.shape[0])


        #num_edges = min(edge_score_map1.size(0), data.edge_index.size(1))
        #edge_index_trimmed = edge_index[:, :num_edges]
        edge_score_map1 = edge_score_map1[:num_edges]
        edge_score_map2 = edge_score_map2[:num_edges]
        edge_score_map3 = edge_score_map3[:num_edges]

        '''
        print(f"edge_score_map1 shape: {edge_score_map1.shape}")
        print(f"edge_score_map2 shape: {edge_score_map2.shape}")
        print(f"edge_score_map3 shape: {edge_score_map3.shape}")
        '''
        
        # Temperature scaling
        edge_score_map1 = F.softmax(edge_score_map1 / self.temperature, dim=-1)
        edge_score_map2 = F.softmax(edge_score_map2 / self.temperature, dim=-1)
        edge_score_map3 = F.softmax(edge_score_map3 / self.temperature, dim=-1)
        
        final_edge_score_map = (edge_score_map1 + edge_score_map2 + edge_score_map3) / 3
        normalized_edge_score_map = self.normalize_scores(final_edge_score_map)
        #print(f"normalized edge_score_map shape: {normalized_edge_score_map.shape}")


        # Convert edge_index and edge scores to NumPy
        
        #final_edge_score_map_np = final_edge_score_map.detach().cpu().numpy()
        #normalized_edge_score_map_np = normalized_edge_score_map.detach().cpu().numpy()

        '''     
        # Debugging: Print sizes
        print(f"in GAT.py edge_index_np shape: {edge_index_np.shape}")
        print("ensure dimensions match")
        print(f"edge_score_map1 trimmed shape: {edge_score_map1.shape}")
        print(f"edge_score_map2 trimmed shape: {edge_score_map2.shape}")
        print(f"edge_score_map3 trimmed shape: {edge_score_map3.shape}")
        print(f"in GAT.py final_edge_score_map_np shape: {final_edge_score_map_np.shape}")
        print(f"in GAT.py normalized_edge_score_map_np shape: {normalized_edge_score_map_np.shape}")
        assert edge_score_map3.size(0) == edge_index.size(1), "Mismatch still exists after trimming!"
        '''


        #num_edges = min(edge_index_np.shape[1], final_edge_score_map_np.shape[0])
        #edge_index_np = edge_index_np[:, :num_edges]  # Trim edge indices
        #final_edge_score_map_np = final_edge_score_map_np[:num_edges]  # Trim scores
        #normalized_edge_score_map_np = normalized_edge_score_map_np[:num_edges]  # Trim normalized scores

        # Combine edges and scores into a single array
        #edges_and_scores = np.hstack((
        #    edge_index_np.T,  # Shape: (num_edges, 2)
        #    final_edge_score_map_np.reshape(-1, 1),  # Unnormalized scores
        #    normalized_edge_score_map_np.reshape(-1, 1)  # Normalized scores
        #))  # Shape: (num_edges, 4)

        # Assume these variables are given:
        # edge_index: Tensor of shape (2, num_edges) containing source and target node indices
        # edge_score_map: Tensor of shape (num_edges,) containing edge scores


        '''
        ########################################################################
        ####this part needs to happen outside of the GAT.py at the very end ####
        ########################################################################
        # Step 1: Convert tensors to NumPy arrays
        edge_index_np = edge_index.detach().cpu().numpy()  # Shape: (2, num_edges)
        final_edge_score_map_np = final_edge_score_map.detach().cpu().numpy()  # Shape: (num_edges,)

        # Step 2: Transpose edge_index to match the shape for concatenation
        edges_transposed = edge_index_np.T  # Shape: (num_edges, 2)

        # Step 3: Reshape edge scores for concatenation
        edge_scores_reshaped = final_edge_score_map_np.reshape(-1, 1)  # Shape: (num_edges, 1)

        # Step 4: Combine edges and scores into a single array
        edges_and_scores = np.hstack((edges_transposed, edge_scores_reshaped))  # Shape: (num_edges, 3)

        # Optionally: Normalize scores (if needed)
        normalized_scores = (final_edge_score_map_np - final_edge_score_map_np.min()) / (final_edge_score_map_np.max() - final_edge_score_map_np.min())
        normalized_scores_reshaped = normalized_scores.reshape(-1, 1)  # Shape: (num_edges, 1)

        # Add normalized scores to the combined array
        edges_and_scores = np.hstack((edges_transposed, edge_scores_reshaped, normalized_scores_reshaped))  # Shape: (num_edges, 4)

        # Step 5: Save to CSV (optional)
        columns = ["Source", "Target", "Score", "NormalizedScore"]
        df = pd.DataFrame(edges_and_scores, columns=columns)
        df.to_csv("edges_with_scores.csv", index=False)

        print("Edges and scores saved to edges_with_scores.csv")


        print(f"BBBBBBBB final edge_score_map shape: {final_edge_score_map_np.shape}")
        # Save to CSV
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)  # Ensure the directory exists
        file_path = os.path.join(results_dir, "edges_with_scores.csv")
        df = pd.DataFrame(edges_and_scores, columns=["Source", "Target", "UnnormalizedScore", "NormalizedScore"])
        df.to_csv(file_path, index=False)

        print(f"Edges with scores saved to {file_path}")
        ########################################################################
        ####this part needs to happen outside of the GAT.py at the very end ####
        ########################################################################
        '''
        return x, edge_score_map1, edge_score_map2, edge_score_map3