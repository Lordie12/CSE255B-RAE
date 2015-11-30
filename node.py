__author__ = 'Lanfear'

from errorfunc import *

class Node:
    """
    Implements the binary tree construction in Python
    """
    def __init__(self, data, is_leaf):
        self.left = None
        self.right = None
        self.data = data
        self.n = 1
        self.n1 = 0
        self.n2 = 0
        self.is_leaf = is_leaf
        self.pb = None
        self.c1bar = None
        self.c1prime = None
        self.c2bar = None
        self.c2prime = None
        self.parent = None
        self.dbar = 0
        self.abar = 0
        self.ebar = 0
        self.pbar = 0
        self.delta = None
        self.gamma = None
        self.eta = None

    def traverse(self, node, W1, W2, Wlabel, d, alpha, t, vocab, Lemb, W1star, W2star, Wlabelstar):
        """
        Compute derivatives at each node, sum them up and return them
        """
        djdx = 0

        thislevel = [node]
        while thislevel:
            nextlevel = list()
            for n in thislevel:
                if n.is_leaf != 1:
                    n.gamma = compute_gamma(n.n1, n.n2, n.ebar, n.c1bar,
                                            n.c1prime, n.c2bar, n.c2prime, alpha, d)

                n.eta = compute_eta(alpha, n.dbar, t)

                if n.parent is None:
                    n.delta = compute_delta_for_internal(n.abar, W1, W2, Wlabel, n.gamma, n.eta, d, 0, 0)

                elif n.is_leaf == 0:
                    if n.parent.left == n:
                        n.delta = compute_delta_for_internal(n.abar, W1[:, 0:d], W2,
                                                                 Wlabel, n.gamma, n.eta, d, n.parent.delta, 1)

                    else:
                        n.delta = compute_delta_for_internal(n.abar, W1[:, d:2 * d], W2,
                                                             Wlabel, n.gamma, n.eta, d, n.parent.delta, 1)

                else:
                    if n.parent.left == n:
                        n.delta = compute_delta_for_internal(0, W1[:, 0:d], W2, Wlabel, None, n.eta, d,
                                                                 n.parent.delta, 2)

                    else:
                        n.delta = compute_delta_for_internal(0, W1[:, d:2 * d], W2, Wlabel, None, n.eta, d,
                                                                 n.parent.delta, 2)

                if n.left:
                    nextlevel.append(n.left)
                if n.right:
                    nextlevel.append(n.right)

                if n.is_leaf != 1:

                    K = np.concatenate((n.c1bar, n.c2bar))
                    L = np.concatenate((K, np.matrix([1]))).T
                    W1star += n.delta * L

                    K = np.concatenate((n.pbar, np.matrix([1]))).T

                    W2star += n.gamma * K

                    Wlabelstar += n.eta * K

                else:
                    djdx = n.delta
                    Lemb[:, vocab[n.data]] += djdx

            thislevel = nextlevel

    def insert_delta(self, data):
        self.data = data

    def insert_dbar_abar_ebar_pbar(self, dbar, abar, ebar, pbar):
        self.dbar = dbar
        self.abar = abar
        self.ebar = ebar
        self.pbar = pbar

    def insert_c1b_c1p_c2b_c2p(self, c1b, c1p, c2b, c2p):
        """
        Function to insert the reconstruction vectors as well as the
        concatenated vector of its children nodes, valid ONLY for internal nodes
        """
        self.c1bar = c1b
        self.c1prime = c1p
        self.c2bar = c2b
        self.c2prime = c2p

    def insert_n1_and_n2(self, n1, n2):
        """
        Function to insert counts of children nodes of current node
        needed for computing future weighted ratios of errors. For e.g.,
        computing gamma in Page 5
        """
        self.n1 = n1
        self.n2 = n2

    def insert_pb_and_n(self, pb, n):
        """
        Function to insert the computed activation, pbar as well
        the sum of counts of its two children + 1, useful for weighted
        reconstruction error calculation
        """
        self.pb = pb
        self.n = n + 1

    def insert_left(self, data):
        self.left = data

    def insert_parent(self, data):
        self.parent = data

    def insert_right(self, data):
        self.right = data

    def get_data(self):
        """
        Function only for leaf nodes, returns the value of the word
        stored in it
        """
        return self.data

    def get_num_nodes(self):
        return self.n

    def is_l(self):
        """
        Function to test if node is a leaf or not
        """
        return self.is_leaf

    def get_pb(self):
        """
        Function which returns outgoing activation pbar
        from an internal node
        """
        return self.pb

    def __str__(self, depth=0):
        """
        Function to print binary tree
        print root calls magic method by default
        """
        ret = ""

        # Print right branch
        if self.right is not None:
            ret += self.right.__str__(depth + 1)


        # Print own value
        ret += '\n' + ('    ' * depth) + str(self.n)

        # Print left branch
        if self.left is not None:
            ret += self.left.__str__(depth + 1)

        return ret