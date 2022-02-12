import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

def labelsort(x,labels):
    """
    Sort label assignments along axes
    """
    
    buff = []
    for clust in np.unique(labels):
        mask = labels == clust
        low_lim = x[mask].min()
        buff.append((clust,low_lim))
    idx = np.stack(buff)
    idx = idx[idx[:,1].argsort()][:,0]
    
    new_lab = np.zeros(len(labels))
    
    count=0
    for i in idx:
        mask = labels == i
        new_lab[mask] = count
        count+=1

    return new_lab

def partition_labels(x,centroids):
    """
    Use centroids to get label assignments
    """
    return np.argmin(
        np.linalg.norm(
            np.expand_dims(x,1) - np.expand_dims(centroids,0),
            axis=-1,ord=2)
        ,axis=1
    )
    

def k_radius(x,centroids):
    """
    Maximal distance between centroids and corresponding samples in partition
    """
    labels = partition_labels(x,centroids)
    radii = []

    for idx in range(centroids.shape[0]):
        mask = labels == idx
        radii.append(
            np.max(
                np.linalg.norm(x[mask]-centroids[idx],axis=-1))
        )
        
    return np.asarray(radii)

def partition_radius(x,centroids):
    """
    ||x1-x2||/2 where x1 and x2 are the two farthest samples in a partition region
    """
    
    labels = partition_labels(x,centroids)
    
    radii = []

    for idx in range(centroids.shape[0]):
        mask = labels == idx
        dist = pairwise_distances(x[mask])
        max_idx = np.argmax(dist.reshape(-1))
        r = max_idx // dist.shape[0]
        c = max_idx - r*dist.shape[0] 
        radii.append(dist[r,c]/2)

    return np.asarray(radii)


def plot_partition_2D(x,centroids,limits=[-6,6],grid_size=1000,
                   s1=50,alpha1=1,s2=100,alpha2=.5,label='Means'):
    
    """
    Plot samples, centroids and partition regions
    -------------------------
    x (nd.array): data samples
    centroids (nd.array): centroids
    limits (list): axis limits
    grid_size (int): number of points per axis
    s1 (float): sample size for plotting
    s2 (float): centroid size for plotting
    alpha1 (float): sample alpha for plotting
    alpha2 (float): centroid alpha for plotting
    label (str): centroid label
    """
    
    dx = np.linspace(limits[0], limits[1], grid_size)
    xx, yy = np.meshgrid(dx, dx)
    Z = partition_labels(np.c_[xx.ravel(), yy.ravel()],centroids)
    Z  = Z.reshape(xx.shape)
    
    fig,ax = plt.subplots()
    
    plt.pcolormesh(xx, yy, Z,cmap='Spectral',shading='gouraud',alpha=0.5)
    ax.scatter(x[:,0],x[:,1],c='w',edgecolors='k',s=30,alpha=alpha1)
    ax.scatter(centroids[:,0],centroids[:,1],c='k',edgecolor='k',marker='X',
                s=s2,alpha=alpha2,label=label)    
    plt.legend()
    return ax