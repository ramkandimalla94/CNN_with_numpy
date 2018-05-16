#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 00:35:00 2018

@author: ram
"""

import numpy as np



""" Reading MNIST data """

'''
==============================================================================================================
'''

from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels

with open('train-images-idx3-ubyte.gz', 'rb') as f:
  train_images = extract_images(f)
  
with open('train-labels-idx1-ubyte.gz', 'rb') as f:
  train_labels = extract_labels(f)
  
with open('t10k-images-idx3-ubyte.gz', 'rb') as f:
  test_images = extract_images(f)
  
with open('t10k-labels-idx1-ubyte.gz', 'rb') as f:
  test_labels = extract_labels(f)
  
'''
==============================================================================================================
'''

#printing shapes of all train and test data

print ("train_images shape = ", train_images.shape)
print ("train_label shape = ",train_labels.shape)
print ("test_images shape = ", test_images.shape)
print ('test_labels shape = ', test_labels.shape)


#padding with Zeros

def padding(x, pad):

     x_pad= np.pad(x,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values=(0,0))
     
     return x_pad
 
    
    
#Convolution
def convolution(a, w, b):
    
    s = np.multiply(a, w) + b

    z = np.sum(s)
    
    z = z+float(b)

    return z


def conv_forward(A_prev, W, b, hparameters):
    
    
  
    # Retrieve dimensions from A_prev shape  
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W shape 
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from hyper parameters 
    stride = hyper_parameters['stride']
    pad = hyper_parameters['pad']
    
    # Convolve dimentions of next layer
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    
    # Initialize the output volume Z with zeros
    Z = np.zeros((m, n_H, n_W, n_C))
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = padding(A_prev, pad)
    
    for i in range(m):                                 # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]                     # Select ith training example padded activation
        for h in range(n_H):                           # loop over vertical axis 
            for w in range(n_W):                       # loop over horizontal axis 
                for c in range(n_C):                   # loop over channels 
                    # Find the corners of the current frame 
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    # Use the corners to define the frame of a_prev_pad
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # Convolve the frame with the correct filter W and bias b, to get back one output neuron
                    Z[i, h, w, c] = convolution(a_slice_prev, W[...,c], b[...,c])
                                        


    # Making sure your output shape is correct
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    # Save information in cache for the backpropagation
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache


np.random.seed(1)
A_prev = np.random.randn(10, 4, 4, 3)
W = np.random.randn(2, 2, 3, 8)
b = np.random.randn(1, 1, 1, 8)
hyper_parameters = {"pad" : 2,
               "stride": 1}

Z, cache_conv = conv_forward(A_prev, W, b, hyper_parameters)

#print Z
#print cache_conv



