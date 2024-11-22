###start imports

import os
import numpy as np
import argparse
import time
import copy
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from imports.ABIDEDataset_cont import ABIDEDataset
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
#from net.GAT import GATNet
from net.GATComplex import GATNet
from imports.utils import train_val_test_split, kfold_train_val_holdout_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from torchsummary import summary
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.preprocessing import StandardScaler
#from important_node_finder_genotype import node_finder
import sys
from datetime import datetime
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.utils import degree
from torch_geometric.utils import remove_self_loops, add_self_loops, coalesce
from torch_geometric.utils import coalesce
import torch_geometric
print(torch_geometric.__version__)
from torch_geometric.data import InMemoryDataset

###end imports


##global init
now = datetime.now()


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
    
    
def graph_statistics(data):
    print(f"Number of edges: {data.edge_index.size(1)}")
    print(f"Graph density: {data.num_edges / (data.num_nodes * (data.num_nodes - 1)) if data.num_nodes > 1 else 0}")


    
####filter graphs ###
def filter_noisy_edges(data, edge_weight_threshold=0.3):
    """
    Filter noisy edges based on edge weights.

    Parameters:
    - data: PyG Data object.
    - edge_weight_threshold: Minimum weight for edges to be retained.

    Returns:
    - data: Updated PyG Data object with filtered edges.
    """
    edge_index, edge_weight = data.edge_index, data.edge_attr

    # Debugging
    print("Edge index shape before filtering:", edge_index.shape)
    print("Edge weight shape before filtering:", edge_weight.shape)

    # Ensure edge weights exist
    if edge_weight is None:
        print("Edge weights not found. Assigning uniform weights.")
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    # Ensure edge_index is in expected shape (2, num_edges)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"Unexpected edge_index shape: {edge_index.shape}")

    # Apply edge weight thresholding
    mask = edge_weight.squeeze() >= edge_weight_threshold
    edge_index = edge_index[:, mask]  # Keep edges with weight above the threshold
    edge_weight = edge_weight[mask]

    # Update the data object
    data.edge_index, data.edge_attr = edge_index, edge_weight

    # Debugging after processing
    print("Edge index shape after filtering:", edge_index.shape)
    print("Edge weight shape after filtering:", edge_weight.shape)

    return data



####end filter graphs    
    
 ###laplacian####

def compute_symmetric_normalized_laplacian(edge_index, num_nodes):
    """
    Compute the symmetric normalized graph Laplacian.

    Parameters:
    - edge_index: Tensor of shape (2, num_edges) representing the edge list.
    - num_nodes: Number of nodes in the graph.

    Returns:
    - laplacian: Tensor of shape (num_nodes, num_nodes) representing the symmetric normalized Laplacian.
    """
    # Convert edge_index to adjacency matrix
    adj_matrix = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

    # Compute degree matrix
    degrees = torch.tensor(adj_matrix.sum(axis=1)).squeeze()
    degree_inv_sqrt = torch.pow(degrees, -0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0.0  # Handle division by zero

    # Compute normalized Laplacian
    D_inv_sqrt = sp.diags(degree_inv_sqrt.numpy())
    identity_matrix = sp.eye(num_nodes)
    normalized_laplacian = identity_matrix - D_inv_sqrt @ adj_matrix @ D_inv_sqrt

    # Convert back to dense tensor
    laplacian = torch.tensor(normalized_laplacian.todense(), dtype=torch.float)
    return laplacian

def transform_features_with_laplacian(data):
    """
    Transform node features using the symmetric normalized Laplacian.

    Parameters:
    - data: PyG Data object.

    Returns:
    - data: Transformed PyG Data object with node features updated.
    """
    laplacian = compute_symmetric_normalized_laplacian(data.edge_index, num_nodes=data.num_nodes)
    data.x = torch.matmul(laplacian, data.x)  # Apply Laplacian to node features
    return data



def preprocessLaplace_dataset(dataset, edge_weight_threshold=0.3):
    """
    Apply Laplacian transformation and filter noisy edges for each graph.

    Parameters:
    - dataset: PyG Data object or list of Data objects.
    - edge_weight_threshold: Minimum weight for edges to be retained.

    Returns:
    - Transformed dataset or single Data object.
    """
    if isinstance(dataset, torch_geometric.data.Data):
        dataset = [dataset]  # Convert to list for uniform processing

    for i in range(len(dataset)):
        print(f"\nGraph {i + 1}: Preprocessing Start")
        print("Before Laplacian Transformation:")
        graph_statistics(dataset[i])
        
        # Apply Laplacian transformation
        dataset[i] = transform_features_with_laplacian(dataset[i])
        
        print("After Laplacian Transformation:")
        graph_statistics(dataset[i])

        # Filter noisy edges
        dataset[i] = filter_noisy_edges(dataset[i], edge_weight_threshold=edge_weight_threshold)
        
        print("After Filtering Noisy Edges:")
        graph_statistics(dataset[i])

    return dataset[0] if len(dataset) == 1 else dataset

#####laplacian #####   
    
    

####define data prep functions





# Normalization function
def normalize_dataset(dataset):
    features = torch.cat([data.x for data in dataset], dim=0).numpy()
    targets = torch.cat([data.y for data in dataset], dim=0).numpy()
    
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Normalize node features
    features = feature_scaler.fit_transform(features)

    # Normalize targets
    targets = target_scaler.fit_transform(targets.reshape(-1, 1)).squeeze()

    # Apply transformations back to each graph
    for data in dataset:
        data.x = torch.tensor(feature_scaler.transform(data.x.numpy()), dtype=torch.float)
        data.y = torch.tensor(target_scaler.transform(data.y.numpy().reshape(-1, 1)).squeeze(), dtype=torch.float)

    return dataset, feature_scaler, target_scaler


####end define data prep functions



##### start run_gnn
def run_gnn(num_sub, num_folds, current_fold, tr_index_arr, te_index_arr, name):

    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", dt_string)
    print('--------full_model_test!!--------')

    torch.manual_seed(123)

    EPS = 1e-10
    device = torch.device("cuda:2")

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=5, help='size of the batches') #default 4
    parser.add_argument('--dataroot', type=str, default='/mnt/newStor/paros/paros_WORK/GAT_alex/ADRC/data/DTI_age_symmetric', help='root directory of the dataset')
    parser.add_argument('--fold', type=int, default=0, help='training which fold')
    parser.add_argument('--lr', type = float, default=0.1, help='learning rate') #default 0.05
    parser.add_argument('--stepsize', type=int, default=30, help='scheduler step size') #default was 50
    parser.add_argument('--gamma', type=float, default=0.25, help='scheduler shrinking rate')
    parser.add_argument('--weightdecay', type=float, default=5e-4, help='regularization') #default=5e-4
    parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
    parser.add_argument('--lamb1', type=float, default=0.1, help='s1 unit regularization')
    parser.add_argument('--lamb2', type=float, default=0.1, help='s2 unit regularization')
    parser.add_argument('--lamb3', type=float, default=0.1, help='s1 entropy regularization')
    parser.add_argument('--lamb4', type=float, default=0.1, help='s2 entropy regularization')
    parser.add_argument('--lamb5', type=float, default=0.1, help='s1 consistence regularization')
    parser.add_argument('--lamb6', type=float, default=0, help='multi-head symmetry regularization')
    parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
    parser.add_argument('--ratio', type=float, default=0.3, help='pooling ratio')
    parser.add_argument('--indim', type=int, default=84, help='feature dim')
    parser.add_argument('--nroi', type=int, default=84, help='num of ROIs')
    parser.add_argument('--nclass', type=int, default=1, help='num of classes')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--temperature', type=float, default=0.3) #default 0.5 #decrease <0/5 if the model is underfitting or the attention scores are too uniform, making them ineffective. Increase >1 to spread attention more evently, but reduce focus
    parser.add_argument('--lambda_entropy', type=float, default=0.001) #defaault 0.001
    parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
    parser.add_argument('--save_path', type=str, default='./model/', help='path to save model')
    opt = parser.parse_args()


    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    #################### Parameter Initialization #######################
    path = opt.dataroot
    save_model = opt.save_model
    #load_model = opt.load_model
    opt_method = opt.optim
    num_epoch = opt.n_epochs
    fold = opt.fold
    writer = SummaryWriter(os.path.join('./log',str(fold)))

    ################## Define Dataloader : Dataset and Model Initialization##################################

    dataset = ABIDEDataset(path,name)
    dataset.data.y = dataset.data.y.squeeze()
    # Normalize dataset
    dataset, feature_scaler, target_scaler = normalize_dataset(dataset)
    #apply laplacian
    
    processed_data = []

    for i, data in enumerate(dataset):
        print(f"Graph {i + 1}: Preprocessing Start")
        print("Before preprocessing:")
        graph_statistics(data)

        # Apply preprocessing
        filtered_data = preprocessLaplace_dataset(data, edge_weight_threshold=0.3)
        processed_data.append(filtered_data)

        print("After preprocessing:")
        graph_statistics(filtered_data)

    # Replace dataset with processed graphs
    class ProcessedDataset(InMemoryDataset):
        def __init__(self, data_list):
            super().__init__()
            self.data, self.slices = self.collate(data_list)

    # Create a new dataset with processed graphs
    dataset = ProcessedDataset(processed_data)
    
    # Graph statistics
    # Iterate through graphs in the dataset to inspect statistics
    
    for idx, data in enumerate(dataset):
        print(f"Graph {idx + 1}:")
    
        # Graph statistics
        print(f"  Number of nodes: {data.num_nodes}")
        print(f"  Number of edges: {data.num_edges}")
        isolated_nodes = data.num_nodes - data.edge_index.unique().size(0)
        print(f"  Isolated nodes: {isolated_nodes}")
        graph_density = data.num_edges / (data.num_nodes * (data.num_nodes - 1)) if data.num_nodes > 1 else 0
        print(f"  Graph density: {graph_density:.4f}")

        # Degree distribution
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        print(f"  Degree distribution: {deg.tolist()}")

        print("-" * 40)
        '''
        # Degree distribution
        deg = degree(data.edge_index[0])
        print(f"Degree distribution: {deg}")
        # Convert PyTorch Geometric graph to NetworkX
        G = to_networkx(data, to_undirected=True)

        # Visualize the graph
        plt.figure(figsize=(10, 10))
        nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
        node_colors = data.x[:, 0].numpy()  # Assuming the first feature is meaningful for coloring
        nx.draw(G, node_color=node_colors, cmap=plt.cm.Blues)
        plt.title("Graph Visualization")
        plt.show()
        '''
    
    # Split into train and validation sets
    tr_index = tr_index_arr.tolist()
    val_index = te_index_arr.tolist()
    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
   
    
    #print('Training index')
    #print(tr_index)
    #print('Testing index')
    #print(val_index)

    
    history_train = np.zeros([4, num_epoch+1])
    model = GATNet(opt.indim,opt.ratio,opt.nclass,opt.batchSize, temperature=opt.temperature,lambda_entropy=opt.lambda_entropy).to(device)
    
    print('model')
    print(model)
    #print('pheno')
    #print(dataset.data.pheno) 
    #print("Pheno Shape Amyloid, pTau, Nfl:", dataset.data.pheno.shape)
    print('fold')
    print(current_fold)
    #print('data loader: data.y')
    #print(dataset.data.y)
    print("data y Shape:", dataset.data.y.shape)

    params = [{'params': model.parameters()}]
   # print("params model :", params)

    if opt_method == 'Adam':
        #optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
        optimizer = torch.optim.Adam(params, lr=opt.lr, weight_decay=opt.weightdecay)
        
    elif opt_method == 'SGD':
        optimizer = torch.optim.SGD(params, lr =opt.lr, momentum = 0.9, weight_decay=opt.weightdecay, nesterov = True)


    optimizer = torch.optim.AdamW(model.parameters(), lr =opt.lr, weight_decay=opt.weightdecay)

    #scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
   
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',          # 'min' because we want to minimize validation loss
        factor=0.5,          # Reduce the LR by half on plateau
        patience=10,          # Number of epochs to wait before reducing LR
        verbose=True,        # Print LR changes
        min_lr=1e-6          # Minimum LR to avoid overly small values
    )
  
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.05, step_size_up=10, mode='triangular2', cycle_momentum=False)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer, base_lr=1e-4, max_lr=1e-2, step_size_up=10, mode="triangular2"
    #)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6) # only for SGD not for Adam
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6, cycle_momentum=False)
   # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    

    ############################### Define Other Loss Functions ########################################
    def topk_loss(s,ratio):
        if ratio > 0.5:
            ratio = 1-ratio
        s = s.sort(dim=1).values
        res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
        return res

 
    def consist_loss(s):
        if len(s) == 0:
            return 0
        s = torch.sigmoid(s)
        W = torch.ones(s.shape[0],s.shape[0])
        D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
        L = D-W
        L = L.to(device)
        res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
        return res


    def symmetric_loss(x):
        batch_size = x.size(0) // opt.indim
        quad_dim = int(opt.indim/2)
        x = x.view(batch_size, opt.indim, opt.indim)
        x = (x - torch.mean(x, dim=(0), keepdim=True)) / (torch.std(x, dim=(0), keepdim=True) + 1e-10)


        q1 = x[:, :quad_dim, :quad_dim]
        q2 = x[:, :quad_dim, quad_dim:]
        q3 = x[:, quad_dim:, quad_dim:]
        q4 = x[:, quad_dim:, :quad_dim]
        mse1 = F.mse_loss(q1, q3)
        mse2 = F.mse_loss(q2, q4)
        return mse1 + mse2




    ###################### Network Training Function: Training Logic#####################################
    def train(epoch, model, train_loader, optimizer):
        """
        Train the model for one epoch.
        """
        #print("Training...")

        for param_group in optimizer.param_groups:
            #print("Learning Rate:", param_group['lr'])

            model.train()  # Set the model to training mode
            loss_all = 0  # Initialize loss accumulator
            s_edge_list = []
            s_edge_1_list = []
            s_edge_2_list = []

        for step, data in enumerate(train_loader):
            if data.x is None or data.edge_index is None:
                continue

            # Move data to the appropriate device
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass
            output, edge_score_map1, edge_score_map2, final_edge_score_map = model(data.x, data.edge_index, data.batch)

            # Pooling
            output = global_mean_pool(output, data.batch)  # Aggregate node-level outputs
            output = output.mean(dim=1, keepdim=True)  # Ensure output matches target shape

            # Compute loss
            attention_scores_list = [edge_score_map1, edge_score_map2, final_edge_score_map]
            loss = model.compute_loss(output.view(-1), data.y.view(-1), attention_scores_list)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            loss_all += loss.item() * data.num_graphs

            # Collect attention scores
            num_edges = min(data.edge_index.size(1), final_edge_score_map.size(0))
            s_edge_list.append(final_edge_score_map[:num_edges].detach().cpu().numpy())
            s_edge_1_list.append(edge_score_map1[:num_edges].detach().cpu().numpy())
            s_edge_2_list.append(edge_score_map2[:num_edges].detach().cpu().numpy())

            #    Debugging (optional)
            #if step % 10 == 0:
            #    print(f"Step {step}, Loss: {loss.item()}")

        # Normalize loss by the number of samples
        loss_all /= len(train_loader.dataset)
        #print(f"Epoch {epoch}, Training Loss: {loss_all}")

        return loss_all, s_edge_1_list, s_edge_2_list, s_edge_list



    ###################### Network Testing Function: Evaluation Logic#####################################
    def test_acc(loader):
        model.eval()
        correct = 0
        y_pred_arr = np.array([])
        y_arr = np.array([])
        for data in loader:
            data = data.to(device)

            output, s_edge_1_list, s_edge_2_list, s_edge_list = model(data.x, data.edge_index, data.batch)

            # Pool output to graph-level predictions
            output = global_mean_pool(output, data.batch)  # [batch_size, 32]
            output = output.mean(dim=1)                   # [batch_size]
            #print(f"Output shape: {output.shape}")
            #print(f"Target shape: {data.y.shape}")

            # Compute differences
            correct += abs(torch.sub(output, data.y)).sum().item()
            # Store predictions and targets
            y_pred_arr = np.concatenate([y_pred_arr, output.detach().cpu().numpy()])
            y_arr = np.concatenate([y_arr, data.y.detach().cpu().numpy()])
    


            #y_pred_arr = np.concatenate([y_pred_arr,output.view(-1).detach().cpu().numpy()])
            # y_arr = np.concatenate([y_arr,data.y.view(-1).detach().cpu().numpy()])
            # correct += abs(torch.sub(torch.squeeze(output),torch.squeeze(data.y))).sum().item()
            
        mae = correct / len(loader.dataset)
        rmse = np.sqrt(np.mean((y_pred_arr - y_arr) ** 2))
        mad_mean = np.mean(np.abs(y_arr - np.mean(y_arr)))
        mad_median = np.mean(np.abs(y_arr - np.median(y_arr)))
        r_squared = 1-(np.sum((y_arr-y_pred_arr)**2))/(np.sum((y_arr-np.mean(y_arr))**2))
        gamma = 1 - (mae/abs(mad_median))

        return mae, rmse, mad_mean, mad_median, r_squared, gamma, y_pred_arr, y_arr

    def test_loss(epoch, model, val_loader):
        """
        Validate or test the model for one epoch.
        """
        model.eval()
        loss_all = 0
        attention_scores = [] 
        with torch.no_grad():  # No gradient computation during evaluation
            for step, data in enumerate(val_loader):
                if data.x is None or data.edge_index is None:
                    continue
                data = data.to(device)
             
                # Forward pass
                try:
                    output, edge_score_map1, edge_score_map2, final_edge_score_map = model(data.x, data.edge_index, data.batch)

                except Exception as e:
                 print(f"Error during model forward pass: {e}")
                 continue
            
                # Ensure the output is valid
                if output is None:
                    print(f"Model output is None at epoch {epoch}.")
                    continue
            
             # Pooling to graph-level representation
                output = global_mean_pool(output, data.batch)  # Graph-level pooling
                output = output.mean(dim=1, keepdim=True)
                # Compute loss
                
                #loss_c = F.mse_loss(output.view(-1), data.y.view(-1))  # Compute loss
                attention_scores_list = [edge_score_map1, edge_score_map2, final_edge_score_map]
                loss = model.compute_loss(output.view(-1), data.y.view(-1), attention_scores_list)
               
                loss_all += loss.item() * data.num_graphs
                attention_scores.append([edge_score_map1, edge_score_map2, final_edge_score_map])
            loss_all /= len(val_loader.dataset)
               


        return loss_all 

    #######################################################################################
    ############################   Model Training #########################################
    #######################################################################################


    

    #tr_index_arr,_,te_index_arr = train_val_test_split(n_sub=num_sub, kfold=num_folds, fold=current_fold)
    
    

    for epoch in range(num_epoch):
        since  = time.time()
        tr_loss, s_edge_1, s_edge_2, s_edge = train(epoch, model, train_loader, optimizer)
        val_loss = test_loss(epoch, model, val_loader)
        scheduler.step(val_loss)
        #for i, edge in enumerate(s_edge):
        #    print(f"Shape of s_edge_1[{i}]: {np.array(edge).shape}")
        #    print(f"Shape of s_edge[{i}]: {np.array(edge).shape}")

        if epoch == num_epoch-1:
          

          #np.save('./results/s_edge_arr_' + name + '_' + str(current_fold) + '.npy', s_edge)
          #print(' ')
            max_len = max(edge.shape[0] for edge in s_edge)
            s_edge_padded = [np.pad(edge, (0, max_len - len(edge)), constant_values=0) for edge in s_edge]
            np.save('./results/s_edge_arr_' + name + '_' + str(current_fold) + '.npy', s_edge_padded)

            #s_edge_normalized = [edge / np.sum(edge) if np.sum(edge) > 0 else edge for edge in s_edge]
            #np.save('./results/s_edge_arr_' + name + '_' + str(current_fold) + '_normalized.npy', s_edge_normalized)
            #print(f"s_edge saved as a normalized NumPy array for fold {current_fold}.")

            #for i, edge in enumerate(s_edge_normalized):
            #    print(f"s_edge_normalized[{i}] size: {len(edge)}")
          
        tr_acc, rmse, mad_mean, mad_median, r_squared, gamma, y_pred, y_arr = test_acc(train_loader)
        val_acc, rmse, mad_mean, mad_median, r_squared, gamma, y_pred, y_arr = test_acc(val_loader)
        #test_accuracy, rmse, mad_mean, mad_median, r_squared, gamma, y_pred, y_arr = test_acc(val_loader)

        
        time_elapsed = time.time() - since
        #print(f'Epoch: {epoch:03d}, Train Loss: {tr_loss:.7f}, Val Loss: {val_loss:.7f}')
        print(f"Epoch {epoch}: Train Loss = {tr_loss}, Validation Loss = {val_loss}")

        #print('*====**')
        #print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        #print('Epoch: {:03d}, Train Loss: {:.7f}, '
        #      'Train Acc: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}'.format(epoch, tr_loss,
        #                                                  tr_acc, val_loss, val_acc))

        writer.add_scalars('Acc',{'train_acc':tr_acc,'val_acc':val_acc},  epoch)
        writer.add_scalars('Loss', {'train_loss': tr_loss, 'val_loss': val_loss},  epoch)
        history_train[0,epoch] = tr_loss
        history_train[1,epoch] = val_loss
        history_train[2,epoch] = tr_acc
        history_train[3,epoch] = val_acc
        #history_train[3,epoch] = test_accuracy


        
        if  epoch == num_epoch-1:
            model_path = os.path.join(opt.save_path,'model_AB_test_' + str(current_fold)+'.pth')
            
            
            best_model_wts = copy.deepcopy(model.state_dict())
            if save_model:
                print('saving model ' + model_path)
                torch.save(best_model_wts,model_path )


    ##saving figures
    import matplotlib.pyplot as plt
    output_path = './results/Loss_AB.png'
    plt.figure(figsize=(10, 6))
    plt.plot(history_train[0, :], label='Train Loss')
    plt.plot(history_train[1, :], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (years)')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.savefig(output_path, dpi=200)
    plt.close()    
    
    
    #Save the figure
    output_path = './results/MAE_AB.png'
    plt.figure(figsize=(10, 6))  # Set figure size
    plt.plot(history_train[2, :], label='Train Error (MAE)')  # Plot training accuracy
    plt.plot(history_train[3, :], label='Test Error (MAE)')  # Plot testing accuracy
    plt.xlabel('Epoch')  # Add x-axis label
    plt.ylabel('MAE(years)')  # Add y-axis label
    plt.title('Mean Absolute Error (Years)')  # Add title
    plt.legend()  # Add legend
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(output_path, dpi=200)
    print(f"Plot saved to {output_path}")
    plt.close()  # Close the plot after saving
        
    
    np.save('./results/train_history_'+ name + '_' + str(current_fold) + '.npy', history_train)
    return len(val_index), tr_acc, val_acc, rmse, mad_mean, mad_median, r_squared, gamma, y_pred, y_arr, name

###########end run_gnn


########Main ######
if __name__ == '__main__':
    num_sub = 65
    num_folds = 5
    num_runs_per_fold = 1  # Number of runs per fold
    all_train_accuracies = []
    all_test_accuracies = []

    base_name = "GAT_ADRC_DTI_age_symmetric_batch_4_lr_0p05_epoch100_dropout_p5_5fold"  # This can be adjusted as per your requirements

    ########################################
    #####         Cross Validation     #####
    ########################################
    for i in range(num_folds):
        fold_train_accuracies = []
        fold_test_accuracies = []
        y_pred_runs = []
        y_true_runs = []

        start_timer = time.perf_counter()

        name = base_name + f"_fold_{i+1}"
        results = np.zeros([8, num_folds])
        train_ind_2, test_ind_2 = train_val_test_split(num_sub, num_folds, i)

        for run in range(num_runs_per_fold):
            # Loop to rerun the logic for the current fold if test accuracy exceeds 100
            while True:
                results[0,1], results[1,1], results[2,1], results[3,1], results[4,1], results[5,1], results[6,1], results[7,1], y_pred, y_arr, _ = run_gnn(num_sub, num_folds, i, train_ind_2, test_ind_2, name)

                train_accuracy = results[1, 1]
                test_accuracy = results[2, 1]

                # If test accuracy is below or equal to 100, break out of the loop
                if train_accuracy <= 200:
                    break

            fold_train_accuracies.append(train_accuracy)
            fold_test_accuracies.append(test_accuracy)
            y_pred_runs.append(y_pred)
            y_true_runs.append(y_arr)
           

        # Compute average accuracies for the current fold
        avg_train_accuracy = sum(fold_train_accuracies) / num_runs_per_fold
        avg_test_accuracy = sum(fold_test_accuracies) / num_runs_per_fold
        all_train_accuracies.append(avg_train_accuracy)
        all_test_accuracies.append(avg_test_accuracy)

        print(f'Fold {i}:')
        print('Average Train accuracy (MAE):', avg_train_accuracy)
        print('Average Test accuracy (MAE):', avg_test_accuracy)

        # Saving results
        np.save('./results/'+name + '_history.npy', results)
        np.save('./results/'+name + '_all_train_acc.npy', fold_train_accuracies)
        np.save('./results/'+name + '_all_test_acc.npy', fold_test_accuracies)

        np.save('./results/'+name + '_all_y_pred.npy', y_pred_runs)
        np.save('./results/'+name + '_all_y_true.npy', y_true_runs)

        end_timer = time.perf_counter()
        dur = end_timer - start_timer
        print('Time elapsed for fold:', dur/60, 'min')
        print('----------')

    # Print all average accuracies after all iterations
    print("Average Train Accuracies (MAE) per Fold:", all_train_accuracies)
    print("Average Test Accuracies (MAE) per Fold:", all_test_accuracies)













