__author__ = 'Haohan Wang'

import numpy as np

def oneHotRepresentation(y, num=10):
    r = []
    for i in range(y.shape[0]):
        l = np.zeros(num)
        l[y[i]] = 1
        r.append(l)
    return np.array(r)


def loadDataCifar10():
    Xtest = np.load('../data/CIFAR10/testData100.npy') / 255.0
    Ytest = np.load('../data/CIFAR10/testLabel100.npy').astype(int)

    return Xtest, oneHotRepresentation(Ytest)

def loadDataCifar10AdvFast(saveName):
    Xadv_fgsm = np.load('../CIFAR10/advs/fgsm' + saveName + '.npy', allow_pickle=True)
    Xadv_pgd = np.load('../CIFAR10/advs/pgd' + saveName + '.npy', allow_pickle=True)

    return Xadv_fgsm, Xadv_pgd

