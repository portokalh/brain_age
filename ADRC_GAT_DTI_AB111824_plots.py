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
from net.GAT import GATNet
from imports.utils import train_val_test_split, kfold_train_val_holdout_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from torchsummary import summary
#from important_node_finder_genotype import node_finder

import sys

import time

from datetime import datetime
now = datetime.now()


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")



def run_gnn(num_sub, num_folds, current_fold, tr_index_arr, te_index_arr, name):

    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", dt_string)
    print('--------full_model_test!!--------')

    torch.manual_seed(123)

    EPS = 1e-10
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    #device = torch.device("cuda")
    device = torch.device("cuda:2")

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='/mnt/newStor/paros/paros_WORK/GAT_alex/ADRC/data/DTI_age_symmetric', help='root directory of the dataset')
    parser.add_argument('--fold', type=int, default=0, help='training which fold')
    parser.add_argument('--lr', type = float, default=0.05, help='learning rate')
    parser.add_argument('--stepsize', type=int, default=50, help='scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.25, help='scheduler shrinking rate')
    parser.add_argument('--weightdecay', type=float, default=5e-4, help='regularization')
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
    parser.add_argument('--save_model', type=bool, default=False)
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

    ################## Define Dataloader ##################################

    dataset = ABIDEDataset(path,name)
    attrs = vars(dataset)
    dataset.data.y = dataset.data.y.squeeze()
    history_train = np.zeros([4, num_epoch+1])

    model = GATNet(opt.indim,opt.ratio,opt.nclass,opt.batchSize).to(device)
    print(model)
    #print(dataset.data.behav)
    print(dataset.data.pheno)
    print(current_fold)


    params = [{'params': model.parameters()}]

    #params = [{'params': model.gnn.parameters()}, {'params': model.gt.parameters(), 'lr': 0.001}, {'params': model.last_layer.parameters(), 'lr': 0.003}] 



    if opt_method == 'Adam':
        #optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
        optimizer = torch.optim.Adam(params, lr=opt.lr, weight_decay=opt.weightdecay)
        
    elif opt_method == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr, momentum = 0.9, weight_decay=opt.weightdecay, nesterov = True)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

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




    ###################### Network Training Function#####################################
    def train(epoch):
        print('train...........')
        scheduler.step()

        for param_group in optimizer.param_groups:
            print("LR", param_group['lr'])
        model.train()
        s0_list = []
        s1_list = []
        s2_list = []
        s0_2_list = []
        cnn_att_list = []
        s_edge_list = []
        s_edge_1_list = []
        s_edge_2_list = []
        loss_all = 0
        step = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            #print(torch.Tensor.size(data.behav))
            output, s_edge_1, s_edge_2, s_edge = model(data.x, data.edge_index, data.batch)
            #print(torch.Tensor.size(output))
            #print(torch.Tensor.size(s_edge))
            #print(s0)
            #s0_list.append(s0.view(-1).detach().cpu().numpy())
            #print(np.size(s0_list))
            #s1_list.append(s1.view(-1).detach().cpu().numpy())
            #s2_list.append(s2.view(-1).detach().cpu().numpy())
            #s0_2_list.append(s0_2.view(-1).detach().cpu().numpy())
            #cnn_att_list.append(cnn_att.view(-1).detach().cpu().numpy())
            s_edge_1_list.append(s_edge.detach().cpu().numpy())
            s_edge_2_list.append(s_edge.detach().cpu().numpy())
            s_edge_list.append(s_edge.detach().cpu().numpy())
            #print(np.size(s_edge.detach().to('cpu').numpy()))


            #loss_c = F.nll_loss(output, data.y)
            #loss_c = F.l1_loss(torch.squeeze(output), torch.squeeze(data.y))
            loss_c = F.mse_loss(torch.squeeze(output), torch.squeeze(data.y))

            #loss_p1 = (torch.norm(w1, p=2)-1) ** 2
            #loss_p2 = (torch.norm(w2, p=2)-1) ** 2
            #loss_tpk1 = topk_loss(s1,opt.ratio)
            #loss_tpk2 = topk_loss(s2,opt.ratio)
            #print(loss_c)
            #print(loss_p1)
            #print(loss_p2)
            #print(loss_tpk1)
            #print(loss_tpk2)
            #loss_sym = symmetric_loss(s_edge)
            loss_consist = 0
            #for c in range(opt.nclass):
            #    loss_consist += consist_loss(s1[data.y == c])
            loss = opt.lamb0*loss_c 
                       #+ opt.lamb6*loss_sym
            writer.add_scalar('train/classification_loss', loss_c, epoch*len(train_loader)+step)

            step = step + 1

            loss.backward(retain_graph=True)
            loss_all += loss.item() * data.num_graphs
            optimizer.step()

        return loss_all / len(train_dataset), s_edge_1_list, s_edge_2_list, s_edge_list


    ###################### Network Testing Function#####################################
    def test_acc(loader):
        model.eval()
        correct = 0
        y_pred_arr = np.array([])
        y_arr = np.array([])
        for data in loader:
            data = data.to(device)
            output, s_edge_1_list, s_edge_2_list, s_edge_list = model(data.x, data.edge_index, data.batch)
            y_pred_arr = np.concatenate([y_pred_arr,output.view(-1).detach().cpu().numpy()])
            y_arr = np.concatenate([y_arr,data.y.view(-1).detach().cpu().numpy()])

            #pred = outputs[0].max(dim=1)[1]
            #print(output)
            #print(data.y)
            #correct += abs(data.y-pred)
            #correct += torch.abs(pred.sub(data.y).sum().item())
            correct += abs(torch.sub(torch.squeeze(output),torch.squeeze(data.y))).sum().item()
            #print(torch.squeeze(output))
            #print(torch.squeeze(data.y))
            #print(correct)
            #correct += pred.eq(data.y).sum().item()
        #print(np.shape(y_pred_arr))
        #print(y_pred_arr)

        #print(np.shape(y_arr))
        #print(y_arr)
        #print('Length of dataset: ')
        #print(len(loader.dataset))
        mae = correct / len(loader.dataset)
        rmse = np.sqrt(np.mean((y_pred_arr - y_arr) ** 2))
        mad_mean = np.mean(np.abs(y_arr - np.mean(y_arr)))
        mad_median = np.mean(np.abs(y_arr - np.median(y_arr)))
        r_squared = 1-(np.sum((y_arr-y_pred_arr)**2))/(np.sum((y_arr-np.mean(y_arr))**2))
        gamma = 1 - (mae/abs(mad_median))

        return mae, rmse, mad_mean, mad_median, r_squared, gamma, y_pred_arr, y_arr

    def test_loss(loader,epoch):
        print('testing...........')
        model.eval()
        loss_all = 0
        for data in loader:
            data = data.to(device)
            output, s_edge_1_list, s_edge_2_list, s_edge_list = model(data.x, data.edge_index, data.batch)
            #loss_c = F.nll_loss(output, data.y)
            #print(output)
            #print(torch.Tensor.size(s_edge))

            loss_c = F.mse_loss(torch.squeeze(output), torch.squeeze(data.y))


            loss = opt.lamb0*loss_c 
                      # + opt.lamb6 * loss_sym

            loss_all += loss.item() * data.num_graphs
        return loss_all / len(loader.dataset)

    #######################################################################################
    ############################   Model Training #########################################
    #######################################################################################




    #tr_index_arr,_,te_index_arr = train_val_test_split(n_sub=num_sub, kfold=num_folds, fold=current_fold)

    tr_index = tr_index_arr.tolist()
    val_index = te_index_arr.tolist()
    #te_index = te_index_arr.tolist()
    #train_dataset = dataset[13:]
    #val_dataset = dataset[0:13]

    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]

    #print(np.shape(val_dataset))
    #test_dataset = dataset[te_index]
    train_loader = DataLoader(train_dataset,batch_size=opt.batchSize, shuffle= True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
    #test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)

    print('Training index')
    print(tr_index)
    print('Testing index')
    print(val_index)

#        best_model_wts = copy.deepcopy(model.state_dict())
#        best_loss = 1e10
    for epoch in range(0, num_epoch):
        #print(epoch)
        since  = time.time()
        tr_loss, s_edge_1, s_edge_2, s_edge = train(epoch)
        if epoch == num_epoch-1:
          #print(torch.Tensor.size(s0_arr))
          #print(np.shape(s0_arr))
          #np.save('s0_arr_' + name + '.npy', s0_arr)
          #np.save('s0_2_arr_' + name + '.npy', s0_2_arr)



          #####np.save('s_edge_arr_' + name + '_' + str(current_fold) + '.npy', s_edge)
          print(' ')



          #print(s0_arr)
        tr_acc, rmse, mad_mean, mad_median, r_squared, gamma, y_pred, y_arr = test_acc(train_loader)
        val_acc, rmse, mad_mean, mad_median, r_squared, gamma, y_pred, y_arr = test_acc(val_loader)
        val_loss = test_loss(val_loader,epoch)
        time_elapsed = time.time() - since
        print('*====**')
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Epoch: {:03d}, Train Loss: {:.7f}, '
              'Train Acc: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}'.format(epoch, tr_loss,
                                                           tr_acc, val_loss, val_acc))

        writer.add_scalars('Acc',{'train_acc':tr_acc,'val_acc':val_acc},  epoch)
        writer.add_scalars('Loss', {'train_loss': tr_loss, 'val_loss': val_loss},  epoch)
        history_train[0,epoch] = tr_loss
        history_train[1,epoch] = val_loss
        history_train[2,epoch] = tr_acc
        history_train[3,epoch] = val_acc


        #print(s1_arr)
        #print(s2_arr)
        #writer.add_histogram('Hist/hist_s1', s1_arr, epoch)
        #writer.add_histogram('Hist/hist_s2', s2_arr, epoch)

# Turning off best model weights


        if  epoch == num_epoch-1:
            print("saving model")
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            if save_model:
                torch.save(best_model_wts, os.path.join(opt.save_path,str(fold)+'.pth'))


    ##saving figures
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(history_train[0, :], label='Train Loss')
    plt.plot(history_train[1, :], label='Test Loss')
    plt.plot(history_train[2, :], label='Train Accuracy (MAE)')
    plt.plot(history_train[3, :], label='Test Accuracy (MAE)')
    plt.savefig('./results/training_testing_losses_accuracies.png', dpi=200)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (MAE)')
    plt.title('Training and Testing Accuracies')
    plt.legend()
    
    plt.figure(figsize=(10, 6))
    plt.plot(history_train[2, :], label='Train Accuracy (MAE)')
    plt.plot(history_train[3, :], label='Test Accuracy (MAE)')
    plt.savefig('./results/MAE.png', dpi=200)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (MAE)')
    plt.title('Training and Testing Error')
    plt.legend()
    
    
    plt.close()


    ###end saving figures            

    #######################################################################################
    ######################### Testing on testing set ######################################
    #######################################################################################

    if opt.load_model:
        model = FAGNN(opt.indim,opt.ratio,opt.nclass).to(device)
        model.load_state_dict(torch.load(os.path.join(opt.save_path,str(fold)+'.pth')))
        model.eval()
        preds = []
        correct = 0
        for data in val_loader:
            data = data.to(device)
            outputs = model(data.x, data.edge_index, data.batch)
            pred = outputs[0].max(1)[1]
            preds.append(pred.cpu().detach().numpy())
            correct += pred.eq(data.y).sum().item()
        preds = np.concatenate(preds,axis=0)
        trues = val_dataset.data.y.cpu().detach().numpy()
        cm = confusion_matrix(trues,preds)
        print("Confusion matrix")
        print(classification_report(trues, preds))

    else:
       model.load_state_dict(best_model_wts)
       model.eval()
       test_accuracy, rmse, mad_mean, mad_median, r_squared, gamma, y_pred, y_arr = test_acc(val_loader)
       test_l= test_loss(val_loader,0)
       print("===========================")
       print("Test Acc: {:.7f}, Test Loss: {:.7f} ".format(test_accuracy, test_l))
       print(opt)
    #kfold_results[0] = len(te_index)
    #kfold_results[1] = test_accuracy
    #print(kfold_results[i,:])

#total_acc = np.sum(kfold_results[0,:]*kfold_results[1,:])/33
#print('Total Test Acc of 10 fold: ' + str(total_acc))
    np.save('./results/train_history_'+ name + '_' + str(current_fold) + '.npy', history_train)
    #top_regions_E2, top_regions_E3, top_regions_E4 = node_finder(name, s0_arr)
    return len(val_index), tr_acc, test_accuracy, rmse, mad_mean, mad_median, r_squared, gamma, y_pred, y_arr, name#, top_regions_E2, top_regions_E3, top_regions_E4, name





if __name__ == '__main__':
    num_sub = 65
    num_folds = 5
    num_runs_per_fold = 1  # Number of runs per fold
    all_train_accuracies = []
    all_test_accuracies = []

    base_name = "GAT_ADRC_DTI_age_symmetric_batch_4_lr_0p05_epoch100_dropout_p5_5fold"  # This can be adjusted as per your requirements

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
           # np.save(name + '_' + str(run+1) + '_y_pred.npy', y_pred)
           # np.save(name + '_' + str(run+1) + '_y_true.npy', y_arr)

        # Compute average accuracies for the current fold
        avg_train_accuracy = sum(fold_train_accuracies) / num_runs_per_fold
        avg_test_accuracy = sum(fold_test_accuracies) / num_runs_per_fold
        all_train_accuracies.append(avg_train_accuracy)
        all_test_accuracies.append(avg_test_accuracy)

        print(f'Fold {i + 1}:')
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













