import sys
path = "/root/"
sys.path.append(path)

import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from models.immodel_d64_h8 import IM_TransferModel
from torch.utils.data import DataLoader
import argparse
from utilspred.dataloader import seqData
from models.bamodel_d64_h5 import BA_Model
from models.elmodel_d64_h8 import EL_Model

def get_args():
    parser = argparse.ArgumentParser(description='The application of baseline',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', dest='input', type=str, default='',
                        help='The input file',metavar='E')
    parser.add_argument('--output', dest='output', type=str, default='',
                        help='The output file',metavar='E')
    return parser.parse_args()


if __name__ == '__main__':
    
    
    
    #Get argument parse
    args = get_args()
    input_file =  args.input
    output_file = args.output

    #Init 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    testDataset = seqData(datapath=input_file) 
    test_loader = DataLoader(testDataset, batch_size=8192, shuffle=False)

    ba_model_dir = '/root/ba/'
    ba_model_basename = 'bamodel_d64_h5_fold*.pth'
    
    el_model_dir = '/root/el/'
    el_model_basename = 'elmodel_d64_h8_fold*.pth'
    
    model_dir = '/root/im/models/'
    model_basename = 'immodel_d64_h8_fold*.pth'
    
    ba_models = [] 
    for a in range(10):
        #
        ba_model = BA_Model().to(device) 

        #
        ba_model_name = ba_model_basename.replace('*', str(a)) 
        ba_model_path = ba_model_dir + ba_model_name 

        ba_weights = torch.load(ba_model_path,map_location=device)
        ba_model.load_state_dict(ba_weights) 

        ba_models.append(ba_model) 

    el_models = [] 
    for b in range(10):
        #
        el_model = EL_Model().to(device) 

        #
        el_model_name = el_model_basename.replace('*', str(b)) 
        el_model_path = el_model_dir + el_model_name 

        el_weights = torch.load(el_model_path,map_location=device)
        el_model.load_state_dict(el_weights) 

        el_models.append(el_model)         
    
    models = [] 
    for n in range(10):
        #
        model = IM_TransferModel().to(device) 

        #
        model_name = model_basename.replace('*', str(n)) 
        model_path = model_dir + model_name 

        weights = torch.load(model_path,map_location=device)
        model.load_state_dict(weights) 


        #model.load_state_dict(torch.load(model_path,map_location=device))
        models.append(model) 

    #Test 
    total_preds_IM = torch.Tensor().to(device) 
    for data in tqdm(test_loader): 

        pep_inputs = data[0].to(device) 
        hla_inputs = data[1].to(device) 

        IM_10M = 0 
        for k in range(10):
            ba_model = ba_models[k]
            el_model = el_models[k]
            model = models[k]
            
            ba_model.eval()
            el_model.eval()
            model.eval() 
            with torch.no_grad(): 
                
                y_ba, ba_hla2pep_attns, ba_pep2hla_attns, ba_hla_enc_self_attns, ba_pep_enc_self_attns, ba_fc1, ba_bn1, ba_re1, ba_dp1, ba_fc2, ba_bn2, ba_re2, ba_outputs = ba_model(hla_inputs, pep_inputs)
                y_el, el_hla2pep_attns, el_pep2hla_attns, el_hla_enc_self_attns, el_pep_enc_self_attns, el_fc1, el_bn1, el_re1, el_dp1, el_fc2, el_bn2, el_re2, el_outputs = el_model(hla_inputs, pep_inputs)      
                
                y_logits, hla2pep_attns, pep2hla_attns, hla_enc_self_attns, pep_enc_self_attns, x_fc1, x_bn1, x_re1, x_dp1, x_fc2, x_bn2, x_re2, outputs = model(hla_inputs, pep_inputs, ba_re1, el_re1)        
                y_IM = torch.sigmoid(y_logits) 

                IM_10M  = IM_10M  + y_IM
                
        ave_IM_10M  = IM_10M  / 10 
        total_preds_IM = torch.cat((total_preds_IM, ave_IM_10M), 0) 

    P_IM = total_preds_IM.detach().cpu().numpy().flatten() 

    #Save to local
    column=['pep','hla','IM score']
    results = list()
    for m in range(len(P_IM)):
        results.append([testDataset.data[m][0],testDataset.data[m][1],P_IM[m]])

    output = pd.DataFrame(columns=column,data=results)
    output.to_csv(output_file,index = None)