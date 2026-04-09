import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

vocab = {'A': 1, 'W': 2, 'V': 3, 'C': 4, 'H': 5, 'T': 6, 'E': 7, 'K': 8, 'N': 9, 'P': 10, 
         'I': 11, 'L': 12, 'S': 13, 'D': 14, 'G': 15, 'Q': 16, 'R': 17, 'Y': 18, 'F': 19, 'M': 20, '-': 0}

class seqData(Dataset):
    def __init__(self, 
                 datapath):
        super(seqData,self).__init__()

        #Load data file
        self.data = pd.read_csv(datapath).values.tolist()
            

    def __len__(self):
        return len(self.data) 

    def __getitem__(self,i):
        pep = self.data[i][0]
        hla = self.data[i][2] #seq
        tgt = self.data[i][3]
        
        
        #input
        pep_inputs = pep.ljust(14, '-')
        pep_inputs = np.asarray([vocab[char] for char in pep_inputs])
        
        hla_inputs = np.asarray([vocab[n] for n in hla])
        
        
        #output
        tgt = float(tgt)
        
        #to tensor
        pep_inputs = torch.LongTensor(pep_inputs)
        hla_inputs = torch.LongTensor(hla_inputs)
        tgt = torch.FloatTensor([tgt])
        
        #return data
        return pep_inputs, hla_inputs, tgt