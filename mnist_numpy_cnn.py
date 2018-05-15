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


def pad(x, pad):

     train_images= np.pad(x,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values=(0,0))
     
     return train_images.shape
 
#pad(train_images,2)

def convolution(a, w, b):
    
    s = np.multiply(a, w) + b

    z = np.sum(s)

    return z


def conv_forward(A_prev, W, b, hparameters):
    
    
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    
    
    ### START CODE HERE ###
    # Retrieve dimensions from A_prev's shape (≈1 line)  
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    
    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m, n_H, n_W, n_C))
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = pad(A_prev, pad)
    
    for i in range(m):                                 # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]                     # Select ith training example's padded activation
        for h in range(n_H):                           # loop over vertical axis of the output volume
            for w in range(n_W):                       # loop over horizontal axis of the output volume
                for c in range(n_C):                   # loop over channels (= #filters) of the output volume
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                    Z[i, h, w, c] = convolution(a_slice_prev, W[...,c], b[...,c])
                                        
    ### END CODE HERE ###

    # Making sure your output shape is correct
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache




