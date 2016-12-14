# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 19:24:02 2016
PythonVersion: 2.7.12
@author: Siyuan Wang

Read image and label data from MNIST hand writing digits data set.
MNIST is downloaded from http://yann.lecun.com/exdb/mnist/

Inspired by https://gist.github.com/akesling/5358964
and https://github.com/sorki/python-mnist

"""
import os
import struct
import numpy as np

class MNIST(object):
    def __init__(self, path = ''):
        self.path = path
        self.train_img_file = 'train-images.idx3-ubyte'
        self.train_lab_file = 'train-labels.idx1-ubyte'
        self.test_img_file = 't10k-images.idx3-ubyte'
        self.test_lab_file = 't10k-labels.idx1-ubyte'
        self.train_img = []
        self.train_lab = []
        self.test_img = []
        self.test_lab = []
        self.rows = 0
        self.columns = 0

    def load(self, tag = 'train'):
        if tag == 'train':
            img_file = os.path.join(self.path, self.train_img_file)
            lab_file = os.path.join(self.path, self.train_lab_file)
        elif tag == 'test':
            img_file = os.path.join(self.path, self.test_img_file)
            lab_file = os.path.join(self.path, self.test_lab_file)
            
        with open(img_file, 'rb') as f:
            magic, num, self.rows, self.columns = struct.unpack('>IIII', f.read(16))
            if magic != 2051:
                raise ValueError('The file "',img_file, 
                '" is not a image file')
            img = np.fromfile(f, dtype = np.uint8).reshape(num, self.rows*self.columns)
            
        with open(lab_file, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            if magic != 2049:
                raise ValueError('The file "', lab_file, 
                '" is not a label file')
            lab = np.fromfile(f, dtype = np.int8)
            
        if tag == 'train':
            self.train_img = img
            self.train_lab = lab
        elif tag == 'test':
            self.test_img = img
            self.test_lab = lab
    
    def display(self, tag='train', num = '0'):
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        
        if tag == 'train':
            image = self.train_img[num].reshape(self.rows, self.columns)
            label = self.train_lab[num]
        
        if tag == 'test':
            image = self.test_img[num].reshape(self.rows, self.columns)
            label = self.test_lab[num]

        plt.imshow(image, cmap=cm.gray)
        plt.title(label)
        plt.show()
