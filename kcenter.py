import numpy as np
from sklearn.metrics import pairwise_distances

class KCenter(object):
    """
    Greedy KCenter algorithm
    """
    def __init__(self,n_clusters, **kwargs):
        self.n_clusters = n_clusters
    
    def fit(self,x,given_centers=np.zeros(0)):
        
        d = pairwise_distances(x)
        n=d.shape[0]

        if self.n_clusters==0:
            cluster_centers = np.array([],dtype=int)
        else:
            if given_centers.size==0:
                cluster_centers = np.random.choice(n,1,replace=False)
                kk = 1
            else:
                cluster_centers = given_centers
                kk = 0

            distance_to_closest = np.amin(d[np.ix_(cluster_centers,np.arange(n))],axis=0)
            while kk<self.n_clusters:
                temp = np.argmax(distance_to_closest)
                cluster_centers = np.append(cluster_centers,temp)
                distance_to_closest = np.amin(np.vstack((distance_to_closest,d[temp,:])),axis=0)
                kk+=1

            cluster_centers = cluster_centers[given_centers.size:]

        self.cluster_centers = cluster_centers
        self.labels_ = self.assign_partition(x,cluster_centers)
        
    def assign_partition(self,x,centers):
        idx = [i for i in range(len(centers))]
        labels = []
        for each in x:
            index = np.argmin(np.linalg.norm(x[centers]-each,axis=1))
            labels.append(idx[index])
        return labels