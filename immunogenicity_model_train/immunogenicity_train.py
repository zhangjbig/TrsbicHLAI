import sys
path = "/root/"
sys.path.append(path)

import numpy as np
import torch
import torch.nn as nn
import os
import random
import math
import pandas as pd
import pyarrow.feather
import time
from models.immodel_d64_h8 import IM_TransferModel
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,recall_score,precision_recall_curve,auc,roc_curve,matthews_corrcoef,confusion_matrix
from utils.dataloader import seqData

from models.bamodel_d64_h5 import BA_Model
from models.elmodel_d64_h8 import EL_Model

def train(model, ba_model, el_model, device, train_loader, criterion, optimizer, epoch): 
    '''
    training function at each epoch
    '''
    #print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    train_loss = 0
    
    logits = torch.Tensor().to(device)
    preds = torch.Tensor().to(device)
    labels = torch.Tensor().to(device)    
    
    for batch_idx, data in enumerate(train_loader):
        
        #get the inputs
        pep_inputs = data[0].to(device)
        hla_inputs = data[1].to(device)

        #zero the parameter gradients
        optimizer.zero_grad()
        
        ######################################### 
        with torch.no_grad():   
            y_ba, ba_hla2pep_attns, ba_pep2hla_attns, ba_hla_enc_self_attns, ba_pep_enc_self_attns, ba_fc1, ba_bn1, ba_re1, ba_dp1, ba_fc2, ba_bn2, ba_re2, ba_outputs = ba_model(hla_inputs, pep_inputs)
            y_el, el_hla2pep_attns, el_pep2hla_attns, el_hla_enc_self_attns, el_pep_enc_self_attns, el_fc1, el_bn1, el_re1, el_dp1, el_fc2, el_bn2, el_re2, el_outputs = el_model(hla_inputs, pep_inputs)      
        ##########################################
        
        #forward
        y_logits, hla2pep_attns, pep2hla_attns, hla_enc_self_attns, pep_enc_self_attns, x_fc1, x_bn1, x_re1, x_dp1, x_fc2, x_bn2, x_re2, outputs = model(hla_inputs, pep_inputs, ba_re1, el_re1) 

        #Calculate loss
        g = data[2].to(device)
        loss = criterion(y_logits.squeeze(), g.squeeze()) 
        train_loss = train_loss + loss.item() 

        #Optimize the model
        loss.backward()
        optimizer.step()
        
        y = torch.sigmoid(y_logits)
            
        logits = torch.cat((logits, y_logits), 0)  
        preds = torch.cat((preds, y), 0)
        labels = torch.cat((labels, g), 0)        


    train_loss_average = train_loss / len(train_loader)
    print("Train Loss Average:",train_loss_average,epoch)
    
    return labels, preds, train_loss_average

def predicting(model, ba_model, el_model, device, loader, criterion, epoch):
    
    model.eval()
    
    val_loss = 0
    
    
    logits = torch.Tensor().to(device)
    preds = torch.Tensor().to(device)
    labels = torch.Tensor().to(device)

    #print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            #Get input
            pep_inputs = data[0].to(device)
            hla_inputs = data[1].to(device)

            #Calculate output
            g = data[2].to(device)
            
            #### 
            y_ba, ba_hla2pep_attns, ba_pep2hla_attns, ba_hla_enc_self_attns, ba_pep_enc_self_attns, ba_fc1, ba_bn1, ba_re1, ba_dp1, ba_fc2, ba_bn2, ba_re2, ba_outputs= ba_model(hla_inputs, pep_inputs)
            y_el, el_hla2pep_attns, el_pep2hla_attns, el_hla_enc_self_attns, el_pep_enc_self_attns, el_fc1, el_bn1, el_re1, el_dp1, el_fc2, el_bn2, el_re2, el_outputs= el_model(hla_inputs, pep_inputs)
            
            #### forward
            y_logits, hla2pep_attns, pep2hla_attns, hla_enc_self_attns, pep_enc_self_attns, x_fc1, x_bn1, x_re1, x_dp1, x_fc2, x_bn2, x_re2, outputs = model(hla_inputs, pep_inputs, ba_re1, el_re1)
                        
            y = torch.sigmoid(y_logits)
            
            logits = torch.cat((logits, y_logits), 0)
            
            preds = torch.cat((preds, y), 0)
            labels = torch.cat((labels, g), 0)
            
            #Calculate loss
            loss = criterion(y_logits.squeeze(), g.squeeze()) 
            val_loss = val_loss + loss.item() 
         
        val_loss_average = val_loss / len(loader)
        print("val Loss Average:",val_loss_average,epoch)
            
    return labels, preds, val_loss_average

def evaluate(tru, pre, epoch): 

    tru_array = tru.detach().cpu().numpy().flatten().astype(np.int64)  
    pre_array = pre.detach().cpu().numpy().flatten().astype(np.float64)  
    
    
    fpr, tpr, thresholds = roc_curve(tru_array, pre_array, pos_label=1)
    area_mine = auc(fpr, tpr)
    
    ####
    AUC_ROC = roc_auc_score(tru_array, pre_array)
    precision_list, recall_list, thresholds_list = precision_recall_curve(tru_array, pre_array, pos_label=1)
    AUC_PR = auc(recall_list, precision_list)

    #### 
    Thresh = 0.5
    pre_binary = [1 if item>=Thresh else 0 for item in pre_array]
    tru_binary = tru_array.tolist()

    accuracy = accuracy_score(tru_binary, pre_binary)
    recall = recall_score(tru_binary, pre_binary) 
    precision = precision_score(tru_binary, pre_binary)
    F1_score = f1_score(tru_binary, pre_binary)
    mcc = matthews_corrcoef(tru_binary, pre_binary)
    
    #
    tn, fp, fn, tp = confusion_matrix(tru_binary, pre_binary).ravel() 
    Specificity = tn / (tn+fp)
    
    #
    NPV = tn / (fn+tn) 
    
    
    evaluation = [accuracy,precision,recall,F1_score,AUC_ROC,AUC_PR,mcc]  
    
    #
    eva_sum = {}
    eva_sum["epoch{}".format(epoch)] = [fpr,tpr,thresholds,area_mine,AUC_ROC,precision_list,recall_list,thresholds_list,AUC_PR,accuracy,recall,precision,F1_score,mcc,Specificity,NPV]
    eva_sum = pd.DataFrame(eva_sum)
    eva_sum = eva_sum.transpose()
    eva_sum.columns = ['fpr', 'tpr', 'thresholds', "area_mine", "AUC_ROC", "precision_list", "recall_list", "thresholds_list", "AUC_PR", "accuracy", "recall", "precision", "F1_score", "mcc", "Specificity", "NPV"]
    
    ### 
    ppv = {}
    ppv["epoch{}".format(epoch)] = [tru_array,pre_array]
    ppv = pd.DataFrame(ppv)
    ppv = ppv.transpose()
    ppv.columns = ['tru_array', 'pre_array']

    return evaluation, eva_sum, ppv

def get_args():
    parser = argparse.ArgumentParser(description='Train the model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fold', dest='fold', type=int, default=0,
                        help='Number of fold',metavar='E')

    return parser.parse_args()

def LoadDataset(fold_num): 
    '''
    Load training dataset and  validation dataset.
    Output:
        trainDataset, valDataset
    '''
    #Load Train and Val Data
    trainDataset = seqData(datapath='/root/trainval_im_fold10/im_train_fold{}.csv'.format(fold_num))
    valDataset = seqData(datapath='/root/trainval_im_fold10/im_val_fold{}.csv'.format(fold_num))

    return trainDataset, valDataset

def seed_torch(seed = 1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def Loss_History(train_cost, val_cost, epoch): 
    
    loss_history = {}
    
    loss_history["epoch{}".format(epoch)] = [train_cost,val_cost]
    loss_history = pd.DataFrame(loss_history)
    loss_history = loss_history.transpose()
    
    loss_history.columns = ['train_loss', 'val_loss']
    
    return loss_history 

if __name__ == '__main__':
    #Fix random seed and reproduce expermental results
    seed_torch()
    #Train setting 
    BATCH_SIZE = 1024
    LR = 1e-5
    NUM_EPOCHS = 200

    #Get argument parse
    args = get_args()


    #Output name
    model_name = 'immodel_d64_h8'

    #fold = 0
    add_name = '_fold' + str(args.fold)
    model_file_name =  '/root/' + model_name + add_name

    #Step 1:Prepare dataloader
    trainDataset, valDataset = LoadDataset(fold_num = args.fold) 
    train_loader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #############################################################################
    #Step 2
    ba_model = BA_Model().to(device)  
    ba_model.load_state_dict(torch.load('/root/ba/bamodel_d64_h5_fold{}.pth'.format(args.fold),map_location=device))
    for name_ba, param_ba in ba_model.named_parameters():
        param_ba.requires_grad = False
    ba_model.eval()  


    el_model = EL_Model().to(device) 
    el_model.load_state_dict(torch.load('/root/el/elmodel_d64_h8_fold{}.pth'.format(args.fold),map_location=device))
    for name_el, param_el in el_model.named_parameters():
        param_el.requires_grad = False
    el_model.eval()
    #############################################################################    
    
    #Step 3: Set  model
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = IM_TransferModel().to(device) 

    #Step 4: construct loss and optimizer
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01) #0.01


    #Step 4: Train                                               
    print(f'''Starting training:
    Epochs:          {NUM_EPOCHS}
    Batch size:      {BATCH_SIZE}
    Learning rate:   {LR}
    Training size:   {len(trainDataset)}
    Validation size: {len(valDataset)}
    Device:          {device.type}
    ''')

    train_evaluation = pd.DataFrame() #train
    train_evaluation_ppv = pd.DataFrame() #train

    val_evaluation = pd.DataFrame()#val
    val_evaluation_ppv = pd.DataFrame()#val
    

    Loss_history  = pd.DataFrame() #loss

    metric_best = -1

    early_stop_count = 0
    
    for epoch in range(NUM_EPOCHS):    
        print("------------epoch:{}".format(epoch))
        
        start_time = time.time()
        print(f"start_time:{start_time}")
        
        #Train
        train_tru, train_pre, train_cost = train(model, ba_model, el_model, device, train_loader, criterion, optimizer, epoch) 
        
        end_time = time.time()
        print(f"end_time:{end_time}")
        execution_time = end_time - start_time
        print(f"execution_time:{execution_time}")

        #Validate
        print('predicting for valid data')
        val_tru, val_pre, val_cost = predicting(model, ba_model, el_model, device, valid_loader, criterion, epoch) 


        #train
        [train_accuracy,train_precision,train_recall,train_F1_score,train_AUC_ROC,train_AUC_PR,train_mcc], train_eva_sum, train_ppv = evaluate(train_tru, train_pre, epoch)
        train_evaluation = pd.concat([train_evaluation, train_eva_sum], axis = 0)  
        train_evaluation_ppv = pd.concat([train_evaluation_ppv, train_ppv], axis = 0)
        

        #val
        [accuracy,precision,recall,F1_score,AUC_ROC,AUC_PR,mcc], val_eva_sum, val_ppv = evaluate(val_tru, val_pre, epoch)
        val_evaluation = pd.concat([val_evaluation, val_eva_sum], axis = 0)  
        val_evaluation_ppv = pd.concat([val_evaluation_ppv, val_ppv], axis = 0)

        

        loss_history = Loss_History(train_cost, val_cost, epoch) 
        Loss_history = pd.concat([Loss_history, loss_history], axis = 0)

        #Logging
        print('TrainEpoch {} accuracy={:.4f},precision={:.4f},recall={:.4f},F1_score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f},mcc={:.4f}'.format(epoch,
    train_accuracy,train_precision,train_recall,train_F1_score,train_AUC_ROC,train_AUC_PR,train_mcc))


        print('ValEpoch {} accuracy={:.4f},precision={:.4f},recall={:.4f},F1_score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f},mcc={:.4f}'.format(epoch,
    accuracy,precision,recall,F1_score,AUC_ROC,AUC_PR,mcc))


        metrics_ep_avg = (AUC_ROC + AUC_PR) / 2
        
        if metric_best < metrics_ep_avg:
            
            metric_best = metrics_ep_avg
            BestEpoch = epoch
            early_stop_count = 0

            #Save model
            torch.save(model.state_dict(), model_file_name +'.pth')

        else:
            early_stop_count = early_stop_count + 1


        print("early_stop_count:{}".format(early_stop_count))
        print('BestEpoch={}; Best Metrics={:.4f}.'.format(
            BestEpoch,metric_best
        ))


        if early_stop_count >= 20:
            print('Early Stop.')
            
            #train_evaluation.to_csv('/root/traineva_fold{}_{}.csv'.format(args.fold,model_name))
            
            #pyarrow.feather.write_feather(val_evaluation,'/root/valeva_fold{}_{}.feather'.format(args.fold,model_name),compression='zstd') 
            #pyarrow.feather.write_feather(Loss_history,'/root/loss_history_fold{}_{}.feather'.format(args.fold,model_name),compression='zstd')
            
            #pyarrow.feather.write_feather(train_evaluation_ppv,'/root/train_nPPV_fold{}_{}.feather'.format(args.fold,model_name),compression='zstd') 
            #pyarrow.feather.write_feather(val_evaluation_ppv,'/root/val_nPPV_fold{}_{}.feather'.format(args.fold,model_name),compression='zstd')
            
            break 