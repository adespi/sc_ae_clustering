#BIC
from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)

"""
from bic import compute_bic
n_clusters = []
BICS = []
for x in range(2,1000,10):
    kmeans = KMeans(n_clusters=x, n_init=20)
    kmeans = kmeans.fit(latent_d)
    bic = compute_bic(kmeans,latent_d)
    print("BIC for {} clusters : {}".format(x, bic))
    n_clusters.append(x)
    BICS.append(bic)

plt.scatter(n_clusters, BICS)
plt.xlabel("k")
plt.ylabel("BIC score")
plt.title("BIC for k in range(2,542,10)")
plt.show()
"""


"""
fig = plt.figure(1)
fig.show()

from sklearn.cluster import KMeans
n_clusters = []
BICS = []
for x in range(2,1000,100):
    kmeans = KMeans(n_clusters=x, n_init=20)
    kmeans = kmeans.fit(latent_d)
    bic = compute_bic(kmeans,latent_d)
    print("BIC for {} clusters : {}".format(x, bic))
    n_clusters.append(x)
    BICS.append(bic)
    plt.scatter(x, bic)

plt.scatter(n_clusters, BICS)
plt.show()
"""


"""
from sklearn.cluster import KMeans
latent_d = np.load("results_brain/ae/latent_d.npy")
n_clusters = []
scores = []


for x in range(2,20,1):
  print(str(x)+" : "+str(int(1.5**x)), end = "\r")
  kmeans = KMeans(n_clusters=int(1.5**x), n_init=4)
  kmeans = kmeans.fit(latent_d[:60000])
  n_clusters.append(int(1.5**x))
  scores.append(kmeans.score(latent_d[:60000]))


import matplotlib.pyplot as plt
plt.scatter(n_clusters, scores)
plt.show()

"""