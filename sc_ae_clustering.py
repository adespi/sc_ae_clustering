"""
Implementation of scDeepCluster for scRNA-seq data
"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

import numpy as np
import pandas as pd

from time import time

from sklearn import metrics
from keras.optimizers import Adam#, SGD
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from model import create_model

#import h5py
import scanpy.api as sc
from preprocess import read_dataset, normalize

from numpy.random import seed
seed(2211)
from tensorflow import set_random_seed
set_random_seed(2211)


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size





if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=7, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='sc_data/snRNA_AD_brain.feather')
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--pretrain_epochs', default=400, type=int)
    parser.add_argument('--gamma', default=1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=0, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--save_dir', default='results')
    parser.add_argument('--model', default='vae')
    parser.add_argument('--model_weight_file', default='weights.h5')
    parser.add_argument('--plot_TNSE', default=True, type=bool)

    args = parser.parse_args()

    # load dataset

    """data_mat = h5py.File(args.data_file)
    x = np.array(data_mat['X'])
    y = np.array(data_mat['Y'])"""
    if ".feather" in args.data_file:
        x = pd.read_feather(args.data_file)#.iloc[:,[x for x in range(250) if x != 98]]
        #gene_names = x.index[np.where(x.values.sum(axis=1)!=0)[0]]
        genes_to_drop = np.where(x.values.sum(axis=1)==0)[0]
        gene_names = pd.read_csv("sc_data/gene_names.csv")['x']
        gene_names = gene_names.drop(genes_to_drop, 0)
        cell_names = x.columns
        x = x.values.T
        y = pd.read_feather("sc_data/colAnnotations.feather").values[:,1]

    elif ".tsv" in args.data_file:
        x=pd.read_csv(args.data_file, sep ='\t')
        gene_names = x.index[np.where(x.values.sum(axis=1)!=0)[0]]
        cell_names = x.columns
        x=x.values.T
        y = np.zeros(x.shape[0])
    #y=np.array(x.columns.str.split('.').tolist())[:,1].astype(np.float)


    # preprocessing scRNA-seq read counts matrix
    adata = sc.AnnData(x)
    adata.obs['Group'] = y

    adata = read_dataset(adata,
                     transpose=False,
                     test_split=False,
                     copy=False)

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    input_size = adata.n_vars

    print(adata.X.shape)
    print(y.shape)

    x_sd = adata.X.std(0)
    x_sd_median = np.median(x_sd)
    print("median of gene sd: %.5f" % x_sd_median)


    if args.update_interval == 0:  # one epoch
        args.update_interval = int(adata.X.shape[0]/args.batch_size)
    print(args)


    # Define scDeepCluster model
    t0 = time()
    print('...Pretraining autoencoder...')

    dims= [input_size, 256, 64, 32]


    model, encoder, imputation_no_zi_network, loss, counts_input, latent_layer = create_model(model = args.model, dims = dims)
    optimizer = Adam(amsgrad=True)

    model.compile(loss= loss, optimizer=optimizer)
    model.summary()
    if args.weights == None:
        es = EarlyStopping(monitor="loss", patience=50, verbose=1)
        model.fit(x=[adata.X, adata.obs.size_factors], y=adata.raw.X, batch_size=args.batch_size, epochs=args.pretrain_epochs, callbacks=[es])
    else:
        assert os.path.isfile(args.weights)
        model.load_weights(args.weights)
    
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.isdir(args.save_dir+"/"+args.model):
        os.mkdir(args.save_dir+"/"+args.model)

    plot_model(model, to_file= args.save_dir+"/"+args.model+"/"+"scDeepCluster_model.pdf", show_shapes=True)
    model.save_weights(args.save_dir+"/"+args.model+"/"+args.model_weight_file)
    print('Pretrained weights are saved to ./' + str(args.save_dir+"/"+args.model+"/"+args.model_weight_file))
    imputed_expression = model.predict(x=[adata.X, adata.obs.size_factors]).astype(np.int)
    imputed_expression_no_zi = imputation_no_zi_network.predict(x=[adata.X, adata.obs.size_factors]).astype(np.int)
    latent_d = encoder.predict(x=adata.X)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(latent_d)
    print("cell clustering finished")
    #np.save(args.save_dir+"/"+args.model+"/"+"latent_d",latent_d)
    pd.DataFrame(latent_d.T, columns = cell_names).to_csv(args.save_dir+"/"+args.model+"/"+"latent_d.csv")
    
    np.save(args.save_dir+"/"+args.model+"/"+"y_pred",y_pred)
    imputed_expression = pd.DataFrame(imputed_expression.T, columns = cell_names, index= gene_names)
    imputed_expression.reset_index(inplace=True)
    imputed_expression.to_feather(args.save_dir+"/"+args.model+"/"+"imputed_expression.feather")
    del imputed_expression
    imputed_expression_no_zi = pd.DataFrame(imputed_expression_no_zi.T, columns = cell_names, index= gene_names)
    imputed_expression_no_zi.reset_index(inplace=True)
    imputed_expression_no_zi.to_feather(args.save_dir+"/"+args.model+"/"+"imputed_expression_no_zi.feather")
    #np.save(args.save_dir+"/"+args.model+"/"+"imputed_expression",imputed_expression)
    print('latent_d values and imputed_expression for each cell are saved to ./' + str(args.save_dir+"/"+args.model+"/"))

    #gene_names = x.index[np.where(x.values.sum(axis=1)!=0)[0]]
    #cell_names = x.columns


    individuals = np.array([int(cell.split(".")[1]) for cell in cell_names.to_list()])

    #means group individuals and cell types
    _ = np.delete(x, genes_to_drop, 1)
    means=np.zeros([len(set(individuals)), len(set(y_pred)), adata.X.shape[1]])
    for cell_type in set(y_pred):
        for individual in (set(individuals)):
            means[individual-1, cell_type] = np.sum(adata.X[(y_pred == cell_type) * (individuals == individual)],axis =0)
    np.save(args.save_dir+"/"+args.model+"/"+'cell_type({})_individuals({})_aggregate'.format(len(set(y_pred)), len(set(individuals))), means.astype(np.int))

    del means

    _ = imputed_expression_no_zi.values[:,1:].T
    del imputed_expression_no_zi
    means_imputed=np.zeros([len(set(individuals)), len(set(y_pred)), adata.X.shape[1]])
    for cell_type in set(y_pred):
        for individual in (set(individuals)):
            means_imputed[individual-1, cell_type] = np.sum(_[(y_pred == cell_type) * (individuals == individual)],axis =0)
    np.save(args.save_dir+"/"+args.model+"/"+'cell_type({})_individuals({})_imputed_aggregate'.format(len(set(y_pred)), len(set(individuals))), means_imputed.astype(np.int))

    del _
    del means_imputed

    gene_names.to_csv(args.save_dir+"/"+args.model+"/"+"gene_names.csv", header = False)
    print('latent_d values and imputed_expression for each cell are saved to ./' + str(args.save_dir+"/"+args.model+"/"))


    if args.plot_TNSE:
        X_embedded = TSNE(n_components=2, verbose=4, early_exaggeration=20, learning_rate=500).fit_transform(latent_d)
        figure1=plt.figure(1)
        plt.scatter(X_embedded[:,0],X_embedded[:,1],c=y_pred,cmap='tab20', s=1)#; plt.show()
        plt.title("Colors represent cluster predictions")
        plt.suptitle("TSNE representation of each cell in the latent dimension of the {} model".format(args.model))
        plt.savefig(args.save_dir+"/"+args.model+"/"+ "TSNE_kmeans.pdf")
        #figure1.show()
    
        figure2=plt.figure(2)
        #colAnnotations = pd.read_feather("sc_data/colAnnotations.feather")
        #for i, cell_type in enumerate(sorted(set(colAnnotations.values[:,1]))):
        for i, cell_type in enumerate(sorted(set(y))):
            corresponding = y == cell_type
            plt.scatter(X_embedded[:,0][corresponding],X_embedded[:,1][corresponding],cmap='tab20', s=1, label=cell_type)
    
    
        plt.title("Colors represent known cell types")
        plt.suptitle("TSNE representation of each cell in the latent dimension of the {} model".format(args.model))
        plt.legend(markerscale=8.)
        plt.savefig(args.save_dir+"/"+args.model+"/"+"TSNE_cell_types.pdf")
        #figure2.show()
        
        print('Figures are saved to ./' + str(args.save_dir+"/"+args.model+"/"))

    y_int = y.copy()
    for i, cell_type in enumerate(sorted(set(y))):
        y_int[y_int==cell_type] = i
    acc = np.round(cluster_acc(y_int, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y_int, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y_int, y_pred), 5)
    print('Final: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
    print('Clustering time: %d seconds.' % int(time() - t0))

    #input("Press Enter to exit...")




"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#X_embedded = TSNE(n_components=2, verbose=4).fit_transform(scDeepCluster.latent_d)
np.save("normal",X_embedded)
X_embedded = TSNE(n_components=2, verbose=4, early_exaggeration=20, learning_rate=500).fit_transform(scDeepCluster.latent_d)
np.save("high",X_embedded)
X_embedded = TSNE(n_components=2, verbose=4, n_iter=5000, early_exaggeration=20, learning_rate=500).fit_transform(scDeepCluster.latent_d)
np.save("high_long",X_embedded)


plt.scatter(X_embedded[:,0],X_embedded[:,1],c=y,cmap='tab20', s=1); plt.show()
plt.scatter(X_embedded[:,0],X_embedded[:,1],c=y_pred,cmap='tab20', s=1); plt.show()

kmeans = KMeans(n_clusters=48, n_init=20) #nclusters : 48
y_pred2 = kmeans.fit(X_embedded).labels_
plt.scatter(X_embedded[:,0],X_embedded[:,1],c=y_pred2,cmap='tab20', s=1); plt.show()

plt.scatter(X_embedded[:,0],X_embedded[:,1], s=1); plt.show()
plt.hist2d(X_embedded[:,0],X_embedded[:,1],(300,300), cmap=plt.cm.jet); plt.show()


"""

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X_embedded = np.load("high.npy")

colAnnotations = pd.read_feather("sc_data/colAnnotations.feather")

#cell_types=sorted(set(colAnnotations.values[:,1]))
#colors = []
#for cell in colAnnotations.values[:,1]:
#  colors.append(cell_types.index(cell))

for i, cell_type in enumerate(sorted(set(colAnnotations.values[:,1]))):
  corresponding = colAnnotations.values[:,1] == cell_type
  colAnnotations.values[:,1][colAnnotations.values[:,1] == cell_type]
  plt.scatter(X_embedded[:,0][corresponding],X_embedded[:,1][corresponding],cmap='tab20', s=1, label=cell_type);

plt.legend(markerscale=10.)
plt.show()


plt.scatter(X_embedded[:,0],X_embedded[:,1],c=colors,cmap='tab20', s=1, label=list(colAnnotations.values[:,1])); plt.legend(); plt.show()
"""




#TODO cluster on latent dimension + plot
#normalizing inverse autoregressive flow
#log var to std
#more epochs