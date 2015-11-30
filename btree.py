__author__ = 'Lanfear'

import numpy as np
from node import *
from errorfunc import softmax

#Infinity value
infty = 9999999


def f(a):
    """
    Activation function tanh (a) as described
    in the project PDF
    """
    return np.tanh(a) / float(np.linalg.norm(a))


def err_rec(c1b, c1p, c2b, c2p, n1, n2):
    """
    Compute reconstruction error using a weighted sum of
    left node reconstruction and right node reconstruction
    """
    k1 = (n1 / float(n1 + n2)) * (np.linalg.norm(c1b - c1p) ** 2)
    k2 = (n2 / float(n1 + n2)) * (np.linalg.norm(c2b - c2p) ** 2)
    return k1 + k2


def compute_binary_tree(line, vocab, L, W1, b1, W2, b2, Wlabel, blabel):
    Nodes = []
    #Intialize all leaf nodes
    for word in line.split():
        try:
            #Check if valid word in vocab
            junk = vocab[word]
            #Second parameter indicates a leaf node
            Nodes.append(Node(word, 1))
        except KeyError:
            pass

    ebar = None

    while len(Nodes) > 1:
        #Intialize minimum error to infinity
        minError = infty
        j = -1
        newNode = Node(None, 0)

        #Loop to find minimum reconstruction error
        for i in range(len(Nodes) - 1):
            #Taking random meaning vector of nodes i and i+1
            #If an input / leaf node, then we return the randomly initialized
            #column vector from L (i.e., meaning vector of d x 1)
            if Nodes[i].is_l() == 1:
                c1bar = L[:, [vocab[Nodes[i].get_data()]]]

            #else we return the column vector previously computed and stored
            else:
                c1bar = Nodes[i].get_pb()

            if Nodes[i+1].is_l() == 1:
                c2bar = L[:, [vocab[Nodes[i + 1].get_data()]]]
            else:
                c2bar = Nodes[i+1].get_pb()

            #Computing vectors a and e, which are two matrices with the products, page 4
            abar = np.add(W1 * np.concatenate((c1bar, c2bar)), b1)

            #Compute activation and reconstruction over these words
            pbar = f(abar)
            pbar /= np.linalg.norm(pbar)
            ebar = np.add(W2 * pbar, b2)
            crec = f(ebar)

            #Taking half matrices for E_rec
            c1prime = crec[0:20, ]
            c2prime = crec[20:40, ]

            #Number of nodes in left and right children
            n1 = Nodes[i].get_num_nodes()
            n2 = Nodes[i+1].get_num_nodes()
            #Computing reconstruction error here
            E_rec = err_rec(c1bar, c1prime, c2bar, c2prime, n1, n2)

            #Update min E_rec keeping track of minimum
            if E_rec < minError:
                minError = E_rec
                j = i
                newNode.insert_pb_and_n(pbar, n1 + n2)
                newNode.insert_n1_and_n2(n1, n2)
                newNode.insert_c1b_c1p_c2b_c2p(c1bar, c1prime, c2bar, c2prime)
                newNode.insert_dbar_abar_ebar_pbar(softmax(Wlabel * pbar + blabel), abar, ebar, pbar)

        #Insert all necessary values into the nodes

        #Update left and right children of newly constructed node
        newNode.insert_left(Nodes[j])
        Nodes[j].insert_parent(newNode)
        newNode.insert_right(Nodes[j+1])
        Nodes[j+1].insert_parent(newNode)

        #Remove my two old nodes and insert the new node
        Nodes.pop(j+1)
        Nodes.pop(j)
        Nodes.insert(j, newNode)

    #Return root node
    return Nodes[0], ebar