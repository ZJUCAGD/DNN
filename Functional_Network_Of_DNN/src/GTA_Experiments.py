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

    # Construct the graph model with the density of 5% of the functional network for GTA
    graph_models = []
    for cor in net_cors:
        graph = Functional_Network(cor=cor, density=0.05)
        graph_models.append(graph)

