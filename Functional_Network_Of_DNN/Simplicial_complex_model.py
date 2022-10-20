import gtda
from gtda.graphs import GraphGeodesicDistance
from gtda.diagrams import PersistenceEntropy,NumberOfPoints,Amplitude
from gtda.homology import VietorisRipsPersistence, SparseRipsPersistence, FlagserPersistence


def computing_PD(cors, homology_dimensions=[0,1]):

    # Construct the simplicial complex sequence based on the cors
    # and compute the persistence diagrams

    
    # matrix pre-processing
    for i in range(len(cors)):
        cor = cors[i]
        cor = abs(cor)
        cor = np.nan_to_num(cor)

        # Since the filtration starts at 0,
        # 1-cor is used as the strength of the functional connectivities
        cor = 1 - cor   

        # To ensure that the network is symmetric
        for ii in range(cor.shape[0]):
            for jj in range(ii):
                cor[ii, jj] = cor[jj, ii]
            cor[ii, ii] = 0
        cors[i] = cor

    # Construct the simplicial complex
    VR = VietorisRipsPersistence(metric="precomputed", homology_dimensions=homology_dimensions)

    # Compute the persistence diagrams
    dgms = VR.fit_transform(cors)
    
    return dgms

def computing_Betti_curve(dgm,delta = 0.01,homology_dimension=0.0):
    
    # Compute the Betti curve

    # dgm: the persistence diagram to be processed
    # delta: the intervals of points on the Betti curve
    # homology_dimension: the dimension of the Betti curve
    
    betti=np.zeros(int(1/delta+1))
    
    # the dgms in a sequence have the same size
    # delete the padded topological features
    if homology_dimension !=0.0:
        dgm = np.unique(dgm, axis=0)
    else:
        betti=betti+1
        
    # Selete the persistence with the corresponding dim 
    index = np.where(dgm[:,2]==homology_dimension)
    persistence = dgm[index][:,0:2]

    # Compute the Betti curve
    for per in persistence:
        b=int(per[0]/0.01)
        d=int(per[1]/0.01)
        betti[b:d]=betti[b:d]+1

    return betti
