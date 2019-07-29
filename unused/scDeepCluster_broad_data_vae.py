"""
Implementation of scDeepCluster for scRNA-seq data
"""

from time import time
import numpy as np
from keras.models import Model
from keras.losses import mse, binary_crossentropy
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input, GaussianNoise, Layer, Activation, Lambda
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping

from sklearn.cluster import KMeans
from sklearn import metrics

import h5py
import pandas as pd
import scanpy.api as sc
from layers import ConstantDispersionLayer, SliceLayer, ColWiseMultLayer
from loss import poisson_loss, NB, ZINB
from preprocess import read_dataset, normalize
import tensorflow as tf

from numpy.random import seed
seed(2211)
from tensorflow import set_random_seed
set_random_seed(2211)

MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)

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


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon







act = 'relu'
#dims= [pd.read_feather("sc_data/snRNA_AD_brain.feather").shape[0]-1, 256, 64, 32] #TODO change dim[0], not correctly defined
dims= [8783, 256, 64, 32] #TODO change dim[0], not correctly defined
#n_clusters=args.n_clusters
noise_sd=2.5
alpha=1.0,
ridge=0,
debug=False
init='glorot_uniform'


n_stacks = len(dims) - 1
# input
x = Input(shape=(dims[0],), name='counts')
h = x
h = GaussianNoise(noise_sd, name='input_noise')(h)
 
# internal layers in encoder
for i in range(n_stacks-1):
    h = Dense(dims[i + 1], kernel_initializer=init, name='encoder_%d' % i)(h)
    h = GaussianNoise(noise_sd, name='noise_%d' % i)(h)    # add Gaussian noise
    h = Activation(act)(h)
# hidden layer
#h = Dense(dims[-1], kernel_initializer=init, name='encoder_hidden')(h)  # hidden layer, features are extracted from here
z_mean = Dense(dims[-1], name='z_mean')(h)
z_log_var = Dense(dims[-1], name='z_log_var')(h)
z = Lambda(sampling, output_shape=(dims[-1],), name='z')([z_mean, z_log_var])

encoder = Model(x, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

latent_inputs = Input(shape=(dims[-1],), name='z_sampling')
sf_layer = Input(shape=(1,), name='size_factors')

# internal layers in decoder
h = latent_inputs
for i in range(n_stacks-1, 0, -1):
    h = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(h)

# output
 
pi = Dense(dims[0], activation='sigmoid', kernel_initializer=init, name='pi')(h)

disp = Dense(dims[0], activation=DispAct, kernel_initializer=init, name='dispersion')(h)

mean = Dense(dims[0], activation=MeanAct, kernel_initializer=init, name='mean')(h)

outputs = ColWiseMultLayer(name='output')([mean, sf_layer])
outputs = SliceLayer(0, name='slice')([outputs, disp, pi])

# instantiate decoder model
decoder = Model([latent_inputs, sf_layer], outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

outputs = decoder([encoder(x)[2], sf_layer])
vae = Model([x, sf_layer], outputs, name='vae_mlp')

plot_model(vae, to_file='vae_mlp.png', show_shapes=True)

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='data.h5')
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--pretrain_epochs', default=400, type=int)
    parser.add_argument('--gamma', default=1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=0, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/scDeepCluster')
    parser.add_argument('--ae_weight_file', default='ae_weights.h5')

    args = parser.parse_args()

    # load dataset
    optimizer1 = Adam(amsgrad=True)
    optimizer2 = 'adadelta'

    """data_mat = h5py.File(args.data_file)
    x = np.array(data_mat['X'])
    y = np.array(data_mat['Y'])"""
    x=pd.read_feather("sc_data/snRNA_AD_brain.feather").iloc[:,:10]
    y=np.array(x.columns.str.split('.').tolist())[:,1].astype(np.float)
    x=x.values.T

    # preprocessing scRNA-seq read counts matrix
    adata = sc.AnnData(x)
    adata.obs['Group'] = y

    adata = read_dataset(adata,
                     transpose=False,
                     test_split=False,
                     copy=True)

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




    #l_pi = vae.get_layer(name='pi').output
    #l_disp = vae.get_layer(name='dispersion').output
    #l_mean = vae.get_layer(name='mean').output
    #zinb = ZINB(pi, theta=disp, ridge_lambda=ridge, debug=debug)
    #reconstruction_loss = zinb.loss(y_true= adata.raw.X, y_pred= output)# tf.get_variable(output, (73909, 17925)))
    #X = tf.Variable([0.0])

    #place = tf.placeholder(tf.float32, shape=(3000000, 300))
    #set_x = X.assign(place)
    
    """reconstruction_loss = binary_crossentropy(y_true= adata.raw.X, y_pred= outputs)# tf.get_variable(output, (17925, 73909)))

    reconstruction_loss *= dims[0]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    #vae_loss = K.mean(kl_loss)"""


    print('...Pretraining autoencoder...')
    def vae_loss(y_pred, y_true):
        reconstruction_loss = binary_crossentropy(y_true= y_true, y_pred= y_pred)# tf.get_variable(output, (17925, 73909)))
        reconstruction_loss *= dims[0]
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss

    vae.compile(loss= vae_loss, optimizer=optimizer1)
    #vae.compile(loss= binary_crossentropy, optimizer=optimizer1)
    vae.summary()
    es = EarlyStopping(monitor="loss", patience=50, verbose=1)
    #vae.fit(x=x, y=y, batch_size=args.batch_size, epochs=args.pretrain_epochs, callbacks=[es])
    vae.fit(x=[adata.X, adata.obs.size_factors], y=adata.raw.X, batch_size=args.batch_size, epochs=args.pretrain_epochs, callbacks=[es])
    vae.save_weights(args.ae_weight_file)
    print('Pretrained weights are saved to ./' + str(args.ae_weight_file))
    pretrained = True

    exit()


    # Pretrain autoencoders before clustering
    if args.ae_weights is None:
        scDeepCluster.pretrain(x=[adata.X, adata.obs.size_factors], y=adata.raw.X, batch_size=args.batch_size, epochs=args.pretrain_epochs, optimizer=optimizer1, ae_file=args.ae_weight_file)

    # begin clustering, time not include pretraining part.

    #scDeepCluster.fit(x_counts=adata.X, sf=adata.obs.size_factors, y=y, raw_counts=adata.raw.X, batch_size=args.batch_size, tol=args.tol, maxiter=args.maxiter,
    #        update_interval=args.update_interval, ae_weights=args.ae_weights, save_dir=args.save_dir, loss_weights=[args.gamma, 1], optimizer=optimizer2)

    # Show the final results
    #scDeepCluster.model.compile(loss=['kld', scDeepCluster.loss], loss_weights=[args.gamma, 1], optimizer=optimizer2)
    #kmeans = KMeans(n_clusters=scDeepCluster.n_clusters, n_init=20)
    #y_pred = kmeans.fit_predict(scDeepCluster.encoder.predict([adata.X, adata.obs.size_factors]))
    #y_pred=scDeepCluster.predict_clusters(adata.X)
    #q, _ = self.model.predict([x_counts, sf], verbose=0)
    #self.y_pred = q.argmax(1)
    #y_pred=scDeepCluster.y_pred
    latent_d = scDeepCluster.encoder.predict([adata.X, adata.obs.size_factors])
    kmeans = KMeans(n_clusters=scDeepCluster.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(latent_d)

    #self.latent_d


    print(y_pred)
    y_pred.size
    acc = np.round(cluster_acc(y, scDeepCluster.y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, scDeepCluster.y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, scDeepCluster.y_pred), 5)
    print('Final: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
    print('Clustering time: %d seconds.' % int(time() - t0))




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
