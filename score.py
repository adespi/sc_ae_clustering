#BIC

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def main():
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--latent_d_file', default='results/vae/latent_d.csv')
    args = parser.parse_args()

    #latent_d = np.load(args.latent_d_file)
    latent_d = pd.read_csv(args.latent_d_file).values[:,1:]
    n_clusters = []
    scores = []


    for x in range(2,20,1):
      print(str(x)+" : "+str(int(1.4**x)), end = "\r")
      kmeans = KMeans(n_clusters=int(1.5**x), n_init=4)
      kmeans = kmeans.fit(latent_d[:int((latent_d.shape[0])*0.85)])
      n_clusters.append(int(1.5**x))
      scores.append(kmeans.score(latent_d[int((latent_d.shape[0])*0.85):]))


    plt.scatter(n_clusters, scores)
    plt.title("K-means score for K")
    plt.xlabel("K")
    plt.ylabel("sklearn K-means score")
    plt.show()


if __name__ == "__main__":
    main()