# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 13:09:11 2021

@author: lpott
"""

import torch.nn as nn
import torch

class FPMC(nn.Module):
    def __init__(self,n_users,n_items,k_UI=64,k_IL=64):
        super(FPMC, self).__init__()
        print("="*10,"Creating FPMC Model","="*10)
        self.n_users = n_users
        self.n_items = n_items
        self.k_UI = k_UI
        self.k_IL = k_IL
        
        self.IL = nn.Embedding(self.n_items,self.k_IL)
        self.LI = nn.Embedding(self.n_items,self.k_IL)
        
        self.UI = nn.Embedding(self.n_users,self.k_UI)
        self.IU = nn.Embedding(self.n_items,self.k_UI)

    def forward(self,uid,basket_prev, iid):
        
        
        x_MF = torch.sum( self.UI(uid) * self.IU(iid) , dim = 1 ) / (self.k_UI**(1/2))
         
        x_FMC = torch.sum( self.IL(iid) * self.LI(basket_prev) ,dim=1 ) / (self.k_IL**(1/2))
        

        return x_MF + x_FMC