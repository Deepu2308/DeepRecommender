# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 11:23:27 2021

@author: deepu
"""

import torch
from torch.utils.data import  DataLoader
from collections import deque
from datetime import datetime

from model    import recommender
from dataprep import Ratings, N_USERS, N_MOVIES

import numpy as np
import os,binascii
import logging


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#script variables
N_EPOCHS          = 100
VALIDATION_WAIT   = 5
BUFFER_SIZE       = 500
model_id          = str(binascii.b2a_hex(os.urandom(5)))[2:-1]

#model hyper parameters
learning_rate, weight_decay = .01,0
train_batch_size, test_batch_size = 200, 1000

print("Loading dataset")
#load train and test
train_dataset = Ratings('train', use_cuda)
test_dataset  = Ratings('test', use_cuda)

#get data loaders
train_loader  = DataLoader(
                               train_dataset, 
                               batch_size=train_batch_size,
                               shuffle=True,
                               num_workers=0
                              )
test_loader   = DataLoader(
                               test_dataset, 
                               batch_size=test_batch_size,
                               shuffle=True,
                               num_workers=0
                              )
test_gen     = iter(test_loader)

#declare model
print("Creating Network")
gen_net   = f"""
net       = recommender(N_USERS, N_MOVIES)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),
                             lr= {learning_rate},
                             weight_decay = {weight_decay})
"""



#start logging
logging.basicConfig(filename=f'src/logs/{model_id}.log',
                    filemode='a', 
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logging.info(gen_net + "\n\n")
logging.info(f"learning_rate : {learning_rate}")
logging.info(f"weight_decay : {weight_decay}")
logging.info(f"max_epochs : {N_EPOCHS}")
logging.info(f"start_time : {datetime.now()}")
logging.info(f"use_cuda  : {use_cuda}\n\n")

#create model
exec(gen_net)
net = net.to(device)
logging.info(str(net) + "\n\n")

test_mse  = deque(maxlen=BUFFER_SIZE)
train_mse = deque(maxlen=BUFFER_SIZE)
best      = np.inf

for epoch in range(N_EPOCHS):
    
    
    for i_batch, train_batch in enumerate(train_loader):
        
        #train
        net.train()
        optimizer.zero_grad()
        target = train_batch['ratings'].reshape((-1,1)).to(device)
        pred   = net(train_batch)
        loss   = criterion(target, pred)
        loss.backward()
        optimizer.step()
        
        train_mse.append(loss.item())
        
        if (i_batch % VALIDATION_WAIT) == 0:
            
            net.eval()
            #get test batch
            try:
                    test_batch      = next(test_gen)
            except StopIteration:
                    test_gen        = iter(test_loader)                
                    test_batch      = next(test_gen)   
                    

            target      = test_batch['ratings'].reshape((-1,1))
            test_pred   = net(test_batch)
            test_loss   = criterion(target, test_pred)
            test_mse.append(loss.item())
            
            mean_train_mse = np.mean(train_mse)
            mean_test_mse  = np.mean(test_mse)
            msg = "epoch: {:5d} batch: {:6d} \t train_Loss: {:4.4f} test_loss: {:4.4f} best_loss: {:4.4f}".format(
                    epoch,i_batch,mean_train_mse,mean_test_mse, best)
            
            #save model
            if best > mean_test_mse :
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
            
                },
                f"src/files/models/{model_id}.pkl")
                
                msg += ' \t Saving Model.'
                best = mean_test_mse
    
            print(msg)
            logging.info(msg)