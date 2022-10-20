import igraph as ig

def Functional_Network(cor, density=0, epsilon = 0.10):
    # 根据cor和epsilon构造二值化功能网络
    # cor是相关系数矩阵
    # weight表示构建加权图还是二值图
    # density表示构建网络是保留的边密度
    # epsilon表示构建二值网络使用的阈值
    # 若density给出，则通过density计算对应的epsilon
    # 若density未给定，则根据给定的epsilon来构造网络
    
    # 网络的构造方法：先根据生成一张加权图G，计算其最大生成树，作为网络的骨干，保证网络的连通性。
    # 之后根据阈值或密度向图中添加边，直到所有满足要求的边被添加入网络。
    
    # 相关系数矩阵预处理
    cor = np.nan_to_num(cor)  # 将nan赋值为0
    cor = abs(cor)            # 消除负权重
    cor_1 = 1-cor               # 计算反权重，用于求最大生成树
    
    G = ig.Graph()
    
    # 消除自连接，并添加顶点
    G.add_vertices(cor.shape[0]) 
    for i in range(cor.shape[0]):
        cor[i,i] = 0
        cor_1[i,i] = 0
 
    Weighted_G = ig.Graph.Weighted_Adjacency(list(cor_1),loops=False, mode ='undirected')
    # 计算最小生成树（即原网络的最大生成树），作为网络的骨干
    mst = Weighted_G.spanning_tree(weights=Weighted_G.es['weight'], return_tree=True)
    mntedgelist = mst.get_edgelist()
    
    # 向G中添加最大生成树中的边
    for e in mntedgelist:
        i,j = e
        cor[i,j]=0
        cor[j,i]=0
        
    G.add_edges(mntedgelist)
        

    # 计算density对应的epsilon
    if density > 0:
        epsilon = np.sort(cor.flatten())[int(cor.shape[0]*cor.shape[0]-(cor.shape[0]*(cor.shape[0]-1)*density-2*cor.shape[0]+2))]
        #print("epsilon={:.6f}".format(epsilon))
    
    # 添加剩余边
    edgelist=[]
    for i in range(cor.shape[0]):
        for j in range(i):
            if cor[i,j]>=epsilon and G.get_eid(i,j,error=False)==-1:
                edgelist.append((i,j))
    G.add_edges(edgelist)
    G=G.simplify()
    
    
    return G 