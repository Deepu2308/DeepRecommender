# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 09:07:04 2021

@author: deepu
"""

import torch
import torch.nn as nn


class recommender(nn.Module):
    """ 
    Model class for deep recommender
    """
    
    def __init__(self, n_users, n_movies):
        super(recommender, self).__init__()
        
        self.user_embedd = nn.Embedding(num_embeddings = n_users, 
                                        embedding_dim = 128, 
                                        #padding_idx: Optional[int]=None, max_norm: Optional[float]=None, 
                                        #norm_type: float=2., scale_grad_by_freq: bool=False, 
                                        #sparse: bool=False
                                        )
        
        self.movie_embedd = nn.Embedding(num_embeddings = n_movies, 
                                        embedding_dim = 128, 
                                        #padding_idx: Optional[int]=None, max_norm: Optional[float]=None, 
                                        #norm_type: float=2., scale_grad_by_freq: bool=False, 
                                        #sparse: bool=False
                                    )
        
        
        self.fc1          = nn.Sequential(
                                        nn.Linear(in_features= 256, 
                                                  out_features=128
                                                  ),
                                        nn.ReLU(),
                                        nn.Dropout(.5)
                                        )
        self.fc2          = nn.Sequential(
                                        nn.Linear(in_features= 128, 
                                                  out_features=64
                                                  ),
                                        nn.ReLU(),
                                        nn.Dropout(.5)
                                        )       
        self.out         = nn.Sequential(
                                        nn.Linear(in_features= 64, 
                                                  out_features=1)
                                        )
        
    def forward(self,input_dict):
        
        users  = input_dict['users']
        movies = input_dict['movies']
        
        users  = self.user_embedd(users)
        movies = self.movie_embedd(movies)
        #print("User : ", users.shape)
        #print("Movie : ", movies.shape)
        
        x     = torch.cat((users,movies), 1)
        #print("x1 : ", x.shape)
        
        x     = self.fc1(x)
        x     = self.fc2(x)
        
        return self.out(x)
        
if __name__ == '__main__':
    
    model = recommender()    
    #model(sample_batched) #can use if you have run data prep from same terminal
        