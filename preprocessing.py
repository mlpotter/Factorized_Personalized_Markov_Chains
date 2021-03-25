# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 13:06:41 2021

@author: lpott
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os

def create_df(filename=None):
    
    print("="*10,"Creating DataFrame","="*10)
    df = pd.read_csv(os.path.join(os.getcwd(),filename),sep='::',header=None)
    df.columns= ['user_id','item_id','rating','timestamp']
    df.sort_values('timestamp',inplace=True)
    
    
    print(df.nunique())
    print(df.shape)
    
    return df.reset_index(drop=True)

class reset_df(object):
    
    def __init__(self):
        print("="*10,"Initialize Reset DataFrame Object","="*10)
        self.item_enc = LabelEncoder()
        self.user_enc = LabelEncoder()
        
    def fit_transform(self,df):
        print("="*10,"Resetting user ids and item ids in DataFrame","="*10)
        df['item_id'] = self.item_enc.fit_transform(df['item_id'])
        df['user_id'] = self.user_enc.fit_transform(df['user_id'])
        
        assert df.user_id.min() == 0
        assert df.item_id.min() == 0 
        
        return df
    
    def inverse_transform(self,df):
        df['item_id'] = self.item_enc.inverse_transform(df['item_id'])
        df['user_id'] = self.user_enc.inverse_transform(df['user_id'])
        return df
    
def create_user_history(df=None):
    if df is None:
        return None
    
    print("="*10,"Creating User Histories","="*10)

    user_history = {}
    for uid in tqdm(df.user_id.unique()):
        user_history[uid] = df[df.user_id == uid].item_id.values.tolist()
            
    return user_history

def train_val_test_split(user_history=None):
    if user_history is None:
        return None
    

    print("="*10,"Splitting User Histories into Train, Validation, and Test Splits","="*10)
    train_history = []
    val_history = []
    test_history = []
    for key,history in tqdm(user_history.items(),position=0, leave=True):
        
        if len(history) < 5:
            print("pop")
        
        for i in np.arange(len(history)-3):
            
            prev_iid,next_iid = history[i:(i+2)]   
            train_history.append((key,prev_iid,next_iid))
                
            
        prev_iid,next_iid = history[(i+1):(i+3)]
        val_history.append((key,prev_iid,next_iid))
        
        prev_iid,next_iid = history[(i+2):(i+4)]
        test_history.append((key,prev_iid,next_iid))
        
    return train_history,val_history,test_history

def create_user_noclick(user_history,df,n_items):
    print("="*10,"Creating User 'no-click' history","="*10)
    user_noclick = {}
    all_items = np.arange(n_items)

    item_counts = df.groupby('item_id',sort='item_id').size()
    #item_counts = (item_counts/item_counts.sum()).values


    for uid,history in tqdm(user_history.items()):
        no_clicks = list(set.difference(set(all_items.tolist()),set(history)))
        item_counts_subset = item_counts[no_clicks]
        probabilities = ( item_counts_subset/item_counts_subset.sum() ).values

        user_noclick[uid] = (no_clicks,probabilities)
        
    return user_noclick
    
    