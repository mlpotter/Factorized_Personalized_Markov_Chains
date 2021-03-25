# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:39:11 2021

@author: lpott
"""
import argparse
from torch.utils.data import DataLoader
import torch

from preprocessing import *
from dataset import *
from metrics import *
from models import *

parser = argparse.ArgumentParser()

parser.add_argument('--num_epochs', type=int, help='Number of Training Epochs', default=250)
parser.add_argument('--alpha', type=float, help='Learning Rate', default=1e-3)
parser.add_argument('--UI_dim',type=int,help="Size of User-Item interaction embedding dimension for matrix factorization",default=64)
parser.add_argument('--IL_dim',type=int,help="Size of Next Item - Last Item interaction embedding dimension for matrix factorization",default=64)
parser.add_argument('--read_filename',type=str,help='The filename to read all the MovieLens-1 million data from to the Dataframe',default="ml-1m\\ratings.dat")
parser.add_argument('--batch_size',type=int,help='The batch size for stochastic gradient descent',default=512)
parser.add_argument('--reg',type=float,help='The regularization strength on l2 norm',default = 0.00005)
parser.add_argument('--hitsat',type=int,help='The number of items to measure the hit@k metric (i.e. hit@10 to see if the correct item is within the top 10 scores)',default=10)
parser.add_argument('--evaluate_every',type=int,help='How frequent (epochs) should you print out the validation and test recall@k metric',default=15)
parser.add_argument('--print_every',type=int,help='How frequent (stochastic gradient updates) should you print out the running training loss',default=250)


# ----------------- Variables ----------------------#


args = parser.parse_args()

read_filename = args.read_filename

num_epochs = args.num_epochs
lr = args.alpha
batch_size = args.batch_size
reg = args.reg

UI_dim = args.UI_dim
IL_dim = args.IL_dim

hitsat = args.hitsat
evaluate_every = args.evaluate_every
print_every = args.print_every 

# ------------------Data Initialization----------------------#

ml_1m = create_df(read_filename)
reset_object = reset_df()
ml_1m = reset_object.fit_transform(ml_1m)

n_users,n_items,n_ratings,n_timestamp = ml_1m.nunique()

user_history = create_user_history(ml_1m)
user_noclicks = create_user_noclick(user_history,ml_1m,n_items)

train_histories,val_histories,test_histories = train_val_test_split(user_history)


# ------------------Model Initialization----------------------#

model = FPMC(n_users,n_items,UI_dim,IL_dim).cuda()
criterion = BPR_Loss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=reg)

s_bpr = BPR_Dataset(train_histories,n_users,n_items)
train_dl = DataLoader(s_bpr,batch_size=batch_size,shuffle=True)

recall_10_val = Recall(val_histories,user_noclicks,n_users,n_items,hitsat)
recall_10_test = Recall(test_histories,user_noclicks,n_users,n_items,hitsat)

max_validation_recall = 0
max_test_recall = 0

# ------------------Training Initialization----------------------#


for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    epoch_loss = 0
    print("="*15 + "Epoch {:d}".format(epoch+1) + "="*15)
    
    model.train()
    for i, data in enumerate(train_dl,0): #enumerate(tqdm(train_dl,position=0, leave=True), 0):
        
        uid,iid_previous,iid_positive,iid_negative = data
                    
            
        iid_previous = iid_previous.cuda()
        uid = uid.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        x_uit = model(uid,iid_previous,iid_positive.cuda())
        x_ujt = model(uid,iid_previous,iid_negative.cuda())
                
        loss = criterion(x_uit, x_ujt)
                
        loss.backward()
        
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if i % print_every == (print_every-1):    # print every 2000 mini-batches
            print('[%d, %5d] Training loss: %.8f' %
                  (epoch + 1, i + 1, running_loss / print_every))

            running_loss = 0.0
        
        epoch_loss += loss.item()
        
        
    if (epoch+1) % evaluate_every == 0:
        rec_val = recall_10_val(model)
        rec_test = recall_10_test(model)
        
        print("Validation Recall: {:.4f}".format(rec_val))
        print("Test Recall: {:.4f}".format(rec_test))

        if max_validation_recall < rec_val:
            max_validation_recall = rec_val
            max_test_recall = rec_test

            
    print("Training Loss: {:.4f}".format(epoch_loss/len(train_dl)))

print('Finished Training')

print('Finished Training')
print("Maximum Validation Hit@{:d} {:.4f}".format(hitsat,max_validation_recall))
print("Maximum Test Hit@{:d} {:.4f}".format(hitsat,max_test_recall))