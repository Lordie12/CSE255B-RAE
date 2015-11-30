__author__ = 'Lanfear'

from scipy.io import loadmat
import pickle
import collections
import btree
from errorfunc import *
import time

if __name__ == '__main__':
    #Length of meaning vector a.k.a 'd'
    d = 20
    #Value of sigma for random sampling of W, b and x_{i} matrices
    sigma = 0.05
    #Number of classes i.e., K = 2 for a binary classifier
    K = 2

    #loads the two polarity data and the vocabulary, stored in vocab.mat
    posText = [x for x in open('data/rt-polarity.pos', 'r')]
    negText = [x for x in open('data/rt-polarity.neg', 'r')]

    #Number of samples, N
    N = float(len(posText) + len(negText))

    #Load vocabulary as a dictionary of (key, value) pairs
    vocab = collections.OrderedDict()
    vocab = {str(k[0]): index for index, k in enumerate(loadmat('data/vocab.mat', )['words'][0][:])}

    #Embedded matrix for vocabulary L = R^{d x |V|}
    L = np.matrix(sigma * np.random.randn(d, len(vocab)))

    '''
    Weight vectors W1, W2 and Wlabel = R^{d x 2d}, R^{2d x d} and R^{K x d}
    W1 represents feedforward, W2 represents reconstruction and Wlabel
    represents classification
    Bias vectors b1, b2 and blabel = R^{d}, R^{2d} and R^{K}
    b1 represents feedforward, b2 represents reconstruction and b3
    represents classification
    alpha represents the weight of reconstruction error and 1-alpha
    represents the weight of classification error
    lambda represents strength of regularization
    '''
    W1 = np.matrix(sigma * np.random.randn(d, 2 * d))
    b1 = np.zeros((d, 1))
    W2 = np.matrix(sigma * np.random.randn(2 * d, d))
    b2 = np.zeros((2 * d, 1))
    Wlabel = np.matrix(sigma * np.random.randn(K, d))
    blabel = np.zeros((K, 1))
    alpha = 0.4
    Lambda = 0.05
    t = np.matrix(([0.5], [0.5]))

    W1star = None
    W2star = None
    Wlabelstar = None
    theta = None

    #Variables to compute back propagation derivatives
    gamma = np.zeros((2 * d, 1))
    xi = 0
    delta = np.zeros((d, 1))
    file = open('theta.txt', 'w')
    file1 = open('vocabmat.txt', 'w')

    for iterc in range(100):

        W1star = np.concatenate((W1, b1), axis=1)
        W2star = np.concatenate((W2, b2), axis=1)
        Wlabelstar = np.concatenate((Wlabel, blabel), axis=1)

        tempmat = np.concatenate((W1star.flatten(), W2star.flatten()), axis=1)
        theta = np.concatenate((tempmat, Wlabelstar.flatten()), axis=1)
        sample = np.linalg.norm(theta)

        start = time.clock()
        totaltime = 0
        print "Iteration %d:" % (iterc + 1)
        count = 0
        #Constructing binary tree for each input sentence
        for line in posText + negText:
            count += 1
            if time.clock() - start >= 20:
                totaltime += time.clock() - start
                start = time.clock()

            #Compute and return my binary tree root
            try:
                [root, ebar] = btree.compute_binary_tree(line, vocab, L, W1, b1, W2, b2, Wlabel, blabel)
            except ValueError:
                continue

            #Softmax distribution stored in dbar
            #dbar = softmax(Wlabel * root.get_pb() + blabel)

            #Create concatenated matrices as per page 4
            W1star = np.concatenate((W1, b1), axis=1)
            W2star = np.concatenate((W2, b2), axis=1)
            Wlabelstar = np.concatenate((Wlabel, blabel), axis=1)

            tempmat = np.concatenate((W1star.flatten(), W2star.flatten()), axis=1)
            theta = np.concatenate((tempmat, Wlabelstar.flatten()), axis=1)

            #Compute all gradients here
            try:
                root.traverse(root, W1, W2, Wlabel, d, alpha, t, vocab, L, W1star, W2star, Wlabelstar)
            except IndexError:
                continue
            except ValueError:
                continue

        print 'Total time elapsed: %d seconds' % totaltime
        #update weights
        #theta = theta - del J / N - lambda * theta
        W1 -= W1star[:, :2 * d] / N - Lambda * W1
        b1 -= W1star[:, 2 * d:2 * d + 1] / N - Lambda * b1

        W2 -= W2star[:, :d] / N - Lambda * W2
        b2 -= W2star[:, d:d + 1] / N - Lambda * b2

        Wlabel -= Wlabelstar[:, :d] / N - Lambda * Wlabel
        blabel -= Wlabelstar[:, d:d + 1] / N - Lambda * blabel

        tempmat = np.concatenate((W1star.flatten(), W2star.flatten()), axis=1)
        theta = np.concatenate((tempmat, Wlabelstar.flatten()), axis=1)
        print np.linalg.norm(theta) - sample

    pickle.dump(theta, file)
    pickle.dump(L, file1)