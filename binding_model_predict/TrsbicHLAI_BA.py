import sys
path = "/root/"
sys.path.append(path)

import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from models.bamodel_d64_h5 import BA_Model
from torch.utils.data import DataLoader
import argparse
from utilspred.dataloader import seqData

def get_args():
    parser = argparse.ArgumentParser(description='The application of baseline',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', dest='input', type=str, default='',
                        help='The input file',metavar='E')
    parser.add_argument('--output', dest='output', type=str, default='',
                        help='The output file',metavar='E')
    return parser.parse_args()


if __name__ == '__main__':
    
    #python predict.py --input ./test.csv --output ./output_test.csv
    
    #Get argument parse
    args = get_args()
    input_file = args.input
    output_file = args.output

    #Init 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    testDataset = seqData(datapath=input_file) 
    test_loader = DataLoader(testDataset, batch_size=8192, shuffle=False)

    model_dir = '/root/ba_models/'
    model_basename = 'bamodel_d64_h5_fold*.pth'

    models = [] 
    for n in range(10):
        #加载模型
        model = BA_Model().to(device) 

        #路径
        model_name = model_basename.replace('*', str(n)) 
        model_path = model_dir + model_name 

        weights = torch.load(model_path,map_location=device)
        model.load_state_dict(weights) 


        #model.load_state_dict(torch.load(model_path,map_location=device))
        models.append(model) 

    #Test 
    total_preds_BA = torch.Tensor().to(device) 
    for data in tqdm(test_loader): 

        pep_inputs = data[0].to(device) 
        hla_inputs = data[1].to(device) 

        BA_10M = 0 
        for model in models: 
            model.eval() 
            with torch.no_grad(): 
                y_logits, hla2pep_attns, pep2hla_attns, hla_enc_self_attns, pep_enc_self_attns, x_fc1, x_bn1, x_re1, x_dp1, x_fc2, x_bn2, x_re2, outputs = model(hla_inputs, pep_inputs)        
                y_BA = torch.sigmoid(y_logits) 

                BA_10M  = BA_10M  + y_BA
                
        ave_BA_10M  = BA_10M  / len(models) 
        total_preds_BA = torch.cat((total_preds_BA, ave_BA_10M), 0) 

    P_BA = total_preds_BA.detach().cpu().numpy().flatten() 

    #Save to local
    column=['pep','hla','BA score']
    results = list()
    for m in range(len(P_BA)):
        results.append([testDataset.data[m][0],testDataset.data[m][1],P_BA[m]])

    output = pd.DataFrame(columns=column,data=results)
    output.to_csv(output_file,index = None)