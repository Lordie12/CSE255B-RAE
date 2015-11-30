__author__ = 'Lanfear'

import numpy as np
import math, sys

def fprime(a):
    """
    Derivative of normalized activation function, sech ^2 (a)
    """
    d = np.shape(a)[0]
    temp = np.diagflat(1 - np.square(np.tanh(a)))
    idmat = np.identity(d) / np.linalg.norm(np.tanh(a))
    idmat -= (np.tanh(a) * np.tanh(a).T) / np.power(np.linalg.norm(np.tanh(a)), 3)

    return temp * idmat


def softmax(weight):
    """
    Return softmax, computed as e^zj / sum(e^zk)
    """
    nr = np.exp(weight)
    return nr / np.sum(nr)


def compute_eta(alpha, dbar, tbar):
    """
    Compute eta, used in chain rule while computing derivative of
    loss function w.r.t. reconstruction
    """
    return (1-alpha) * (dbar - tbar)


def compute_gamma(n1, n2, ebar, c1bar, c1prime, c2bar, c2prime, alpha, d):
    """
    Compute gamma, used in chain rule while computing derivative of
    loss function w.r.t. classification
    """
    mat1 = -2 * alpha * fprime(ebar)
    mat2 = np.concatenate((n1 / float(n1 + n2) * (c1bar - c1prime), n2 / float(n1 + n2) * (c2bar - c2prime)))
    return mat1 * mat2


def compute_delta_for_internal(abar, W1, W2, Wlabel, gamma, eta, d, deltaq, nodetype):
    """
    Compute total delta for internal nodes
    """
    #temp1 computes total contribution of all nodes' parent error
    #temp2 computes total contribution of all nodes' reco. error
    #temp3 computes total contribution of all nodes' label error

    #Root node delta
    temp1 = None
    temp2 = None
    if gamma is not None:
        temp2 = W2.T * gamma
    temp3 = Wlabel.T * eta
    if nodetype == 0:
        return fprime(abar) * (temp2 + temp3)

    #Internal node delta
    elif nodetype == 1:
        temp1 = W1.T * deltaq
        return fprime(abar) * (temp1 + temp2 + temp3)

    #Leaf node delta
    else:
        temp1 = W1.T * deltaq
        return temp1 + temp3


def compute_delta_for_leaf(abar, W1, W2, Wlabel, gamma, eta, deltaq = 0):
    """
    Compute total delta for leaf nodes
    """
    #temp1 computes total contribution of all nodes' parent error
    #temp2 computes total contribution of all nodes' label error
    temp1 = np.transpose(W1) * deltaq
    temp2 = np.transpose(Wlabel) * eta
    return temp1 + temp2