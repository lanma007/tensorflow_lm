#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 22:14:55 2018

@author: lanma
"""

import numpy as np
import pandas as pd
import os

class Input_data:

    def __init__(self, batch_size, n_step_encoder, n_step_decoder):                                   
        # read the data 
        #data = pd.read_csv('./toydata.csv')
        os.chdir('/Users/lanma/Documents/dissertation/Attention-RNN-code/DA_RNN/data')
        data = pd.read_csv('./47banks_ori.csv')
        #data = pd.read_csv('./47banks_diffK.csv')
        self.data = np.array(data)
       
        #self.t=5
        #self.t=20
        #self.t=60        
        #data=np.array(data)
        #data1=data[:,:47]
        #data2=data[:,94:141]#t=5
        #data2=data[:,141:188]#t=20
        #data2=data[:,188:235]#t=60
        #data=np.concatenate((data1, data2), axis=1)
        #t=data.shape[0]
        #self.data=data[:t-4,:]
        #self.data=data[:t-19,:]
        #self.data=data[:t-59,:]

        
        #self.data = df.iloc[:,:17].values
                
        #self.train = self.data[:3500, :]#2000-2014
        #self.val = self.data[3500:,:]#2005-2014
        #self.test = self.data[3500:,:]
        
        self.train = self.data[:2000, :]#2000-2007
        self.val = self.data[2000:,:]
        #shuffle training set
        #self.train= random.sample(self.train,len(self.train))
        #self.train = np.array(self.train)
        # parameters for the network                 
        self.batch_size = batch_size
        self.n_step_encoder = n_step_encoder
        self.n_step_decoder = n_step_decoder

        self.n_train = len(self.train)
        self.n_val = len(self.val)
        self.n_feature = self.data.shape[1]
        self.n_label = self.data.shape[1]

        
        # data normalization
#        self.stdev = np.std(self.train,axis=0)
        # in case the stdev=0,then we will get nan
#        for i in range (len(self.stdev)):
#            if self.stdev[i]<0.00000001:
#                self.stdev[i] = 1
#        self.train = (self.train-self.mean)/self.stdev
#        self.test = (self.test-self.mean)/self.stdev
#        self.val = (self.val - self.mean)/self.stdev
                   
    def batch_norm_wrapper(self,inputs):

        batch_mean = np.mean(inputs,0)
        batch_var = np.var(inputs,0)
        return (inputs-batch_mean)/np.sqrt(batch_var+1e-7)                   


    def training(self):  
        index = np.arange(0, self.n_train-self.n_step_encoder-self.n_step_decoder+1)
        index_size = len(index)
        train_x = np.zeros([index_size, self.n_step_encoder, self.n_feature])
        train_label = np.zeros([index_size, self.n_step_decoder,self.n_label])
        temp = 0
        for item in index:
            train_x[temp,:,:] = self.train[item:item + self.n_step_encoder, :self.n_feature]
            train_label[temp,:,:] = self.train[item+self.n_step_encoder:item+self.n_step_encoder+self.n_step_decoder, :self.n_feature]
            temp += 1
        train_x=self.batch_norm_wrapper(train_x)
        train_label=self.batch_norm_wrapper(train_label)        
        return train_x, train_label

        
    def validation(self):
        index = np.arange(0, self.n_val-self.n_step_encoder-self.n_step_decoder+1)
        index_size = len(index)
        val_x = np.zeros([index_size, self.n_step_encoder, self.n_feature])
        val_label = np.zeros([index_size, self.n_step_decoder,self.n_label])
        temp = 0
        for item in index:
            val_x[temp,:,:] = self.val[item:item + self.n_step_encoder, :self.n_feature]
            val_label[temp,:,:] = self.train[item+self.n_step_encoder:item+self.n_step_encoder+self.n_step_decoder, :self.n_feature]
            temp += 1
        val_x=self.batch_norm_wrapper(val_x)
        val_label=self.batch_norm_wrapper(val_label)        
        return val_x, val_label
    

 