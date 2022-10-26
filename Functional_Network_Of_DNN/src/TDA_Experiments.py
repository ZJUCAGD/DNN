import numpy as np

from Model import FCN, train
from Graph_model import Functional_Network
from Simplicial_complex_model import computing_PD

if __name__ == '__main__':

    # the path to save the results
    root = "../results/MNIST_[100,50]/"
    reg_dirs = ['Vanilla', 'Dropout', 'BatchNorm']
    
    # Load the saved cor matrix
    net_cors = []
    for d in reg_dirs:
        for i in range(net_nums):
            net_cors.append(np.loadtxt(root+d+"/net_cor/" + str(i) + 'th_FNN.txt'))

    # Construct the simplicial complex model of the functional network for TDA
    net_dgms = computing_PD(net_cors, homology_dimensions=[0,1])

