__author__ = 'Haohan Wang'

import foolbox

import numpy as np

import time


def attackAModelCIFAR(x, logits, maximumPixel, XTest, yTest):
    start = time.time()
    yTest = np.argmax(yTest, 1)

    fmodel = foolbox.models.TensorFlowModel(x, logits, (0, maximumPixel))
    # fmodel = foolbox.models.TensorFlowModel(x, logits, (-124.0, maximumPixel-104.0)) # for ImageNet

    attack_fgsm = foolbox.attacks.FGSM(fmodel)
    attack_pgd = foolbox.attacks.ProjectedGradientDescentAttack(fmodel, distance=foolbox.distances.Linfinity)

    Xadv_fgsm = np.zeros_like(XTest)
    Xadv_pgd = np.zeros_like(XTest)


    for i in range(XTest.shape[0]):

        img = XTest[i].reshape([32, 32, 3])

        print '\r processed ', i+1, '/', XTest.shape[0], (time.time()-start)/60.0, 'minutes passed',

        result = attack_fgsm(img, yTest[i])
        if result is not None:
            Xadv_fgsm[i] = result
        else:
            Xadv_fgsm[i] = img

        result = attack_pgd(img, yTest[i])
        if result is not None:
            Xadv_pgd[i] = result
        else:
            Xadv_pgd[i] = img


    print()
    print('Totally cost', (time.time()-start)/3600.0, 'hours')


    return Xadv_fgsm, Xadv_pgd

