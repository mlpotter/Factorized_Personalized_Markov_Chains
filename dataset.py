# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 13:07:19 2021

@author: lpott
"""


from torch.utils.data import Dataset
import torch
import numpy as np

class BPR_Dataset(Dataset):
    """Movie Lens User Dataset"""
    
    def __init__(self,train_tuples,n_users,n_items):
        print("="*10,"Creating PyTorch Dataset Object","="*10)
        self.train_tuples = np.array(train_tuples)
        self.n_users = n_users
        self.n_items = n_items        
        self.n_tuples = len(train_tuples)
        
        
    def __len__(self):
        return self.n_tuples
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()  
            
        uid,iid_prev,iid_next = self.train_tuples[idx]                
        
        while True:
            iid_negative = np.random.randint(0,self.n_items)
            #if len(set.intersection(set(iid_negative),set([iid_next]))) == 0:
            #    break
            if iid_negative != iid_next:
                break
        
        sample = (uid,iid_prev,iid_next,iid_negative)
        
        return sample