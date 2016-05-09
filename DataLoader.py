import numpy as np
import sys
import os
from sklearn.feature_extraction import image

__author__ = 'Paul'


def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict


def iterate_cifar(shapeInput, batch_size, shuffle=False, train=True):
    # iterator over patches of the cifar10 data set.
    files = []
    if train:
        for j in range(1, 6):
            files.append('data_batch_'+str(j))
    else:
        for j in range(1, 6):
            files.append('test_batch')
    data_idxs = np.random.permutation(len(files))
    data = []
    labels = []
    for j in range(len(files)):
        data_idx = j
        if shuffle:
            data_idx = data_idxs[j]
        file = files[data_idx]
        dict = unpickle('C:\\Paul\\cifar-10-batches-py\\'+file)
        ls = dict['labels']
        idxs = np.random.permutation(len(dict['data']))
        for i in range(len(dict['data'])):
            if shuffle:
                idx = idxs[i]
            else:
                idx = i
            stackedArray = np.dstack((dict['data'][idx][0:1024].reshape(32, 32),
                                      dict['data'][idx][1024:1024 * 2].reshape(32,32),
                                      dict['data'][idx][1024 * 2:1024 * 3].reshape(32, 32)))
            patches = image.extract_patches_2d(stackedArray, (shapeInput[0], shapeInput[1]), max_patches=1)
            #max = patches.max()+1.e-6
            patches = patches.astype(np.float32) / 256.0
            data.append(patches)
            labels.append(ls[idx])
            if len(data)>=batch_size:
                array = np.asarray(data).reshape(-1, shapeInput[0]*shapeInput[1]*3)
                data = []
                labels = []
                #print(len(dict['data'])*len(files)*patches.shape[0])
                yield array