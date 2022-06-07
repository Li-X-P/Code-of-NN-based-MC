# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 13:41:35 2021

@author: xiaopenli4
"""
import opt_einsum as oe
import itertools
import scipy.io
import numpy as np
import torch

def loaddata(data):
    data = scipy.io.loadmat(data)
    return data

def onehot(n, index):
    '''
    :param n: Dimension number.
    :param index: Index list.
    :return: Return n-dimension one hot vectors. (len(index) * n)
    '''
    index = torch.unsqueeze(index, dim=1)
    onehot = torch.zeros(len(index), n).scatter_(1, index, 1)
    return onehot


def batch_full_onehot(tensor_shape, index_list):
    batch_size = index_list.shape[0]
    output = []
    for i in range(index_list.shape[1]):

        index =index_list[:,i]
        index = torch.unsqueeze(index, dim=1)
        onehot = torch.zeros(batch_size,tensor_shape[i]).scatter_(1, index, 1)
        output.append(onehot)
    return output


def generateIndex_list(Omega):
    index = np.where(Omega == 1)
    row = index[0]
    col = index[1]
    index_list = []
    for i in range(len(row)):
        index_list.append([row[i],col[i]])
    return index_list 