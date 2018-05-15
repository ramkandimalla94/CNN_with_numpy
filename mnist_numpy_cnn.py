#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 00:35:00 2018

@author: ram
"""

import numpy as np
import pandas as pd

from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels

with open('/Users/ram/Downloads/stuff/balayya/mnist_numpy_cnn/train-images-idx3-ubyte.gz', 'rb') as f:
  train_images = extract_images(f)
  
with open('/Users/ram/Downloads/stuff/balayya/mnist_numpy_cnn/train-labels-idx1-ubyte.gz', 'rb') as f:
  train_labels = extract_labels(f)
  
with open('/Users/ram/Downloads/stuff/balayya/mnist_numpy_cnn/t10k-images-idx3-ubyte.gz', 'rb') as f:
  test_images = extract_images(f)
  
with open('/Users/ram/Downloads/stuff/balayya/mnist_numpy_cnn/t10k-labels-idx1-ubyte.gz', 'rb') as f:
  test_labels = extract_labels(f)
  

  
  

  
