import igraph as ig

def Functional_Network(cor, density=0):
    # Construct the functional network based on the cors matrix

    # cor: the cor matrix of neurons
    # density: the density of the constructed functional neetwork
    
    
    # matrix pre-processing
    cor = np.nan_to_num(cor)
    cor = abs(cor)
    
    # In order to generate a maximum spanning tree
    cor_1 = 1-cor
    
    G = ig.Graph()
    
    # Eliminate the self-loop, and add the vertices
    G.add_vertices(cor.shape[0]) 
    for i in range(cor.shape[0]):
        cor[i,i] = 0
        cor_1[i,i] = 0
 
    Weighted_G = ig.Graph.Weighted_Adjacency(list(cor_1),loops=False, mode ='undirected')
    # The minimum spanning tree
    #(i.e., the maximum spanning tree of the original network)
    # is calculated as the backbone of the network
    mst = Weighted_G.spanning_tree(weights=Weighted_G.es['weight'], return_tree=True)
    mntedgelist = mst.get_edgelist()
    
    # Add edges in the maximum spanning tree to G
    for e in mntedgelist:
        i,j = e
        cor[i,j]=0
        cor[j,i]=0
        
    G.add_edges(mntedgelist)
        

    # Calculate the epsilon corresponding to the density
    if density > 0:
        epsilon = np.sort(cor.flatten())[int(cor.shape[0]*cor.shape[0]-(cor.shape[0]*(cor.shape[0]-1)*density-2*cor.shape[0]+2))]
        
    # Add remaining edges    
    edgelist=[]
    for i in range(cor.shape[0]):
        for j in range(i):
            if cor[i,j]>=epsilon and G.get_eid(i,j,error=False)==-1:
                edgelist.append((i,j))
    G.add_edges(edgelist)
    G=G.simplify()
    
    
    return G 
