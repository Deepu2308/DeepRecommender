# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 23:02:55 2021

@author: deepu
"""
from dataprep import Ratings, N_USERS, N_MOVIES
from model  import recommender
from torch.utils.data import  DataLoader
from os import mkdir
from os.path  import isdir

import pandas as pd
import numpy as np
import torch
import seaborn as sns
import argparse


def plot(model_id):
    
    #make model plot folder if it doesnt exist
    if not isdir(f'src/plots/{model_id}/'):
        mkdir(f'src/plots/{model_id}/')
        
    
    #load model
    print("Loading model")
    net = recommender(N_USERS, N_MOVIES)
    model = torch.load(f"src/files/models/{model_id}.pkl")
    net.load_state_dict(model['model_state_dict'])
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    net.to(device)
    net.eval()
    
    #load data
    print("Loading data")
    test_dataset  = Ratings('test', use_cuda)
    test_loader   = DataLoader(
                                   test_dataset, 
                                   batch_size=1000,
                                   shuffle=True,
                                   num_workers=0
                                  )
    train = pd.read_csv('input/splits/train.csv')
    
    #predict
    labels, users, movies = [], [], []
    predictions = []
    n_iter = 10
    print("Collecting prediction samples")
    for i_batch, sample_batched in enumerate(test_loader):   
        
        #collect info
        users_       = sample_batched['ratings'].cpu().numpy()
        movies_      = sample_batched['ratings'].cpu().numpy()
        rating_       = sample_batched['ratings'].cpu().numpy()
        
        #store info
        labels.append(rating_)
        users.append(users_)
        movies.append(movies_)
        
        #predict
        pred        = net(sample_batched).detach().cpu().numpy().flatten()
        predictions.append(pred)
    
        #exit    
        print(f"{i_batch + 1}/{n_iter}")
        if (i_batch+1) % n_iter == 0: break
        
    #make prediciton dataframe    
    predictions = np.array(predictions).flatten()
    labels      = np.array(labels).flatten()
    users       = np.array(users).flatten()
    movies      = np.array(movies).flatten()
    
    print("Analysing Output")
    df = pd.DataFrame({
        'actual_rating' : labels,
        'predicted_rating' : predictions,
        'user' : users,
        'movie': movies
        })
    
    #order by rating
    df.sort_values('actual_rating', inplace = True)
    
    def pred_plotter(df, comment = ''):
        #plot
        sns.set_theme(style="whitegrid")
        pred_plot = sns.violinplot( y ='predicted_rating',
                                    x ='actual_rating',
                                    data = df,
                                    palette="Set3",
                                    dodge=False)
        pred_plot.set_title('Actual Ratings vs Predicted Rating' + ('' if comment == '' else  f'({comment})'))
        
        pred_plot.figure.savefig(f'src/plots/{model_id}/pred_plot{comment}.png')
    
        return pred_plot
    
    #plot all
    pred_plotter(df, comment = '')
    
    #plot seen
    pred_plotter(df[df.user.isin(train.userId.unique())],
                 comment = '_seen_users'
                 )
    
    def hist_plotter(df, comment = ''):
    
        n = df.shape[0]
        temp = pd.DataFrame(
            {
                'Rating' : list(df.actual_rating)  + list(df.predicted_rating),
                'Type'   : ['Actual']*n  + ['Predicted']*n
                }        
            )
        hist_plot = sns.histplot(x = 'Rating', hue = 'Type', data = temp)
        
        hist_plot.set_title('Actual Ratings vs Predicted Rating Histogram'+ ('' if comment == '' else  f'({comment})'))
        
        hist_plot.figure.savefig(f'src/plots/{model_id}/hist_plot{comment}.png')
        
        return hist_plot
    
    #plot hist of all
    hist_plotter(df)
    
    #plot hist of seen users
    hist_plotter(df[df.user.isin(train.userId.unique())],
                 comment= '_seen_users')
    
    print("Plots available at " + f'src/plots/{model_id}/')

if __name__ == '__main__':
    
    #collect args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        default = '32d18d7d5c',
                        help = 'pass the model id'
                       )
    
    args = parser.parse_args()
    
    #call plot function
    plot(args.m)