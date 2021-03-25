# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 13:08:25 2021

@author: lpott
"""

import torch
import numpy as np

class BPR_Loss(object):
    def __init__(self):    
        print("="*10,"Creating Model Loss Criterion (s-BPR)","="*10)
        pass
        
    def __call__(self,positive_scores,negative_scores):
        return torch.log( 1 + torch.exp(-positive_scores+negative_scores) ).sum()
    
class Recall(object):
    def __init__(self,user_history,user_noclick,n_users,n_items,k=10):
        print("="*10,"Creating Hit@{:d} Metric Object".format(k),"="*10)

        self.user_history = user_history
        self.user_noclick = user_noclick
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
                

    def __call__(self,model):
        model.eval()
        with torch.no_grad():
            
            scores = np.zeros(self.n_items)
            running_recall = 0 
            
            data = torch.LongTensor(self.user_history).cuda()
            
            
            
            for row in data:#tqdm(data,position=0, leave=True):
                
                negatives,probabilities = self.user_noclick[row[0].cpu().item()]
                
                uid = row[0].repeat(101)
                
                prev_iid = row[1].repeat(101)
                next_iid = row[2]          
                
                sampled_negatives = np.random.choice(negatives,size=100,replace=False,p=probabilities).tolist() + [next_iid.cpu().item()]
                sampled_negatives = torch.LongTensor(sampled_negatives).cuda()
                
                scores = model(uid,prev_iid,sampled_negatives)
                
                
                                    
                top_items = torch.argsort(scores,descending=True)[:self.k]
                
                running_recall += torch.sum(top_items == 100).cpu().item()
                
        return running_recall/self.n_users * 100
