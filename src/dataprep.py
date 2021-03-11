# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 08:44:17 2021

@author: deepu
"""

import torch
import joblib

import numpy  as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder


N_USERS  = 283228
N_MOVIES = 53889

class Ratings(Dataset):
    """Movielens rating dataset."""

    def __init__(self,  
                 file, # 'train' or 'test'
                 use_cuda = False
                 ):
        """
        Args:
            file (string, optional):     'train' or 'test'
        """
        
        self.df     = pd.read_csv(f'input/splits/{file}.csv')
        self.device = torch.device("cuda" if use_cuda else "cpu")
        
        #load encoders
        self.user_enc      = joblib.load('src/files/user_enc.pkl')
        self.movie_enc     = joblib.load('src/files/movie_enc.pkl')
    
        #convert to tensor
        self.users         = torch.Tensor(self.user_enc.transform(self.df.userId.values))
        self.movies        = torch.Tensor(self.movie_enc.transform(self.df.movieId.values))
        self.ratings       = torch.Tensor(self.df.rating.values)
        
    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = {'users': self.users[idx].long().to(self.device), 
                  'movies': self.movies[idx].long().to(self.device), 
                  'ratings': self.ratings[idx].to(self.device) 
        }
        
        return sample


    
if __name__ == '__main__':
    
    #read data
    p_input = 'input/ratings.csv'
    ratings = pd.read_csv(p_input)    
    #ratings = ratings.sample(1000)

    #number of unique users and movies    
    n_users = ratings.userId.nunique()
    n_movies= ratings.movieId.nunique()    
    print("Number of users: " , n_users)
    print("Number of movies: ", n_movies)
    
    #create user encoder
    user_enc  = LabelEncoder()
    user_enc.fit(ratings.userId.values)
    
    #create movie encoder
    movie_enc = LabelEncoder()
    movie_enc.fit(ratings.movieId.values)
    
    #save encoder objects
    joblib.dump(user_enc,'src/files/user_enc.pkl')
    joblib.dump(movie_enc,'src/files/movie_enc.pkl')
    
    #train test split
    train,test= train_test_split(ratings)
    train.to_csv('input/splits/train.csv')
    test.to_csv('input/splits/test.csv')
    
    #example usage for Ratings class
    train_dataset = Ratings(file  = 'train')
    
    train_loader  = DataLoader(
                               train_dataset, 
                               batch_size=4,
                               shuffle=True,
                               num_workers=0
                              )
    
    
    for i_batch, sample_batched in enumerate(train_loader):
        print(i_batch, sample_batched['users'].size(),
              sample_batched['movies'].size(),
              sample_batched['ratings'].size())
    