"""
Implementation of scDeepCluster for scRNA-seq data
"""

from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from keras.models import Model
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input, Lambda, GaussianNoise, Layer, Activation
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from keras.activations import softmax

from sklearn.cluster import KMeans
from sklearn import metrics

import h5py
import scanpy.api as sc
from layers import ConstantDispersionLayer, SliceLayer, ColWiseMultLayer, Concatenate
from loss import poisson_loss, NB, ZINB, KL
from preprocess import read_dataset, normalize
import tensorflow as tf

from numpy.random import seed
seed(2211)
from tensorflow import set_random_seed
set_random_seed(2211)

from tensorflow import ConfigProto
from keras.backend.tensorflow_backend import set_session
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

#TODO change
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)
temperature = K.variable(5.0,name="temperature")

def sampling(logits_y):
    #ins = K.variable(1.0,name="gf") #Input((1,))
    #U = Lambda(lambda x: K.random_uniform(K.shape(logits_y), 0, 1)*x, name = "gumbel_layer")(ins)
    U = K.random_uniform(K.shape(logits_y), 0, 1)
    y = logits_y - K.log(-K.log(U + 1e-20) + 1e-20) # logits + gumbel noise
    y = softmax( y / temperature)
    return y

#z = Lambda(sampling, output_shape=(M*N,))(logits_y)

def sample_gumbel(logits, eps=1e-20): 
  """Sample from Gumbel(0, 1)"""
  ##U = Lambda(K.random_uniform)([shape, K.variable(0, dtype='int32'), K.variable(1, dtype='int32')])
  ##return -K.log(-K.log(U + eps) + eps)
  U = Lambda(K.random_uniform, arguments={'minval':0,'maxval':1, 'dtype':tf.float32})(K.shape(logits))
  #U = K.random_uniform(K.shape(logits),minval=0,maxval=1, dtype= tf.float32)
  return -K.log(-K.log(U + eps) + eps)

def gumbel_softmax_sample(logits): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  #z = Lambda(sampling)(logits)
  #Lambda(sample_gumbel)(logits)
  #y = logits + Lambda(sample_gumbel)(logits)
  y = sampling(logits)
  #y = Lambda(sampling, name = "gumbel_layer")(logits)
  #return softmax( y / temperature)
  return y

def gumbel_softmax(logits, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits)
  if hard:
    y_hard = K.cast(K.equal(y,K.max(y,1,keepdims=True)),y.dtype)
    y = K.stop_gradient(y_hard - y) + y
  return y

def sampling_kld(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


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


def autoencoder(dims, n_clusters, noise_sd=0, init='glorot_uniform', act='relu', temp=500.0):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        Model of autoencoder
    """
    n_stacks = len(dims) - 1
    # input
    sf_layer = Input(shape=(1,), name='size_factors')
    x = Input(shape=(dims[0],), name='counts')
    h = x
    #h = GaussianNoise(noise_sd, name='input_noise')(h)

    # internal layers in encoder
    for i in range(n_stacks):
        h = Dense(dims[i + 1], kernel_initializer=init, name='encoder_%d' % i)(h)
        #h = GaussianNoise(noise_sd, name='noise_%d' % i)(h)    # add Gaussian noise
        h = Activation(act)(h)
    # hidden layer
    #h = Dense(n_clusters, kernel_initializer=init, name='encoder_hidden')(h)  # hidden layer, features are extracted from here
    #K.set_value(tau, np.max([K.get_value(tau) * np.exp(- anneal_rate * e), min_temperature]))
    #global temperature
    #temperature = K.variable(temp,name="temperature")

    z_mean = Dense(n_clusters, name='z_mean')(h)
    z_log_var = Dense(n_clusters, name='z_log_var')(h)

    z_output = Concatenate([z_mean, z_log_var], axis=-1)
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling_kld, output_shape=(n_clusters,), name='z')([z_mean, z_log_var])
    
    #gumbel_layer = Lambda(gumbel_softmax, name = "gumbel_layer", arguments={'hard':False})(h)
    #h = gumbel_softmax(h,hard=False)
    #h=gumbel_layer
    h=z
    # internal layers in decoder
    for i in range(n_stacks, 0, -1):
        h = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(h)

    # output
 
    pi = Dense(dims[0], activation='sigmoid', kernel_initializer=init, name='pi')(h)

    disp = Dense(dims[0], activation=DispAct, kernel_initializer=init, name='dispersion')(h)

    mean = Dense(dims[0], activation=MeanAct, kernel_initializer=init, name='mean')(h)

    output = ColWiseMultLayer(name='output')([mean, sf_layer])
    output = SliceLayer(0, name='slice')([output, disp, pi])

    return Model(inputs=[x, sf_layer], outputs=[output, z_output])


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class SCDeepCluster(object):
    adata=""
    def __init__(self,
                 dims,
                 n_clusters=10,
                 noise_sd=0,
                 alpha=1.0,
                 ridge=0,
                 debug=False,
                 temp=500.0):

        super(SCDeepCluster, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.noise_sd = noise_sd
        self.alpha = alpha
        self.act = 'relu'
        self.ridge = ridge
        self.debug = debug
        self.autoencoder = autoencoder(self.dims, self.n_clusters, noise_sd=self.noise_sd, act = self.act, temp = temp)
        
        # prepare clean encode model without Gaussian noise
        ae_layers = [l for l in self.autoencoder.layers]
        hidden = self.autoencoder.input[0]
        for i in range(1, len(ae_layers)):
            if "noise" in ae_layers[i].name:
                next
            elif "dropout" in ae_layers[i].name:
                next
            else:
                hidden = ae_layers[i](hidden)
            #if "encoder_hidden" in ae_layers[i].name:  # only get encoder layers
            if "z" in ae_layers[i].name:  # only get encoder layers
                break
        self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)

        pi = self.autoencoder.get_layer(name='pi').output
        disp = self.autoencoder.get_layer(name='dispersion').output
        mean = self.autoencoder.get_layer(name='mean').output
        zinb = ZINB(pi, theta=disp, ridge_lambda=self.ridge, debug=self.debug)
        self.loss = zinb.loss

        clustering_layer = ClusteringLayer(self.n_clusters, alpha=self.alpha, name='clustering')(hidden)
        self.model = Model(inputs=[self.autoencoder.input[0], self.autoencoder.input[1]],
                           outputs=[clustering_layer]+ self.autoencoder.output)

        self.pretrained = False
        self.centers = []
        self.y_pred = []


    def cluster_acc_2(self, y_true, y_pred):
        data_mat = h5py.File("10X_PBMC.h5")
        y = np.array(data_mat['Y'])
        #adata=xex
        global adata
        y_prediction = self.encoder.predict(x=[adata.X, adata.obs.size_factors]).argmax(1)
        #y_prediction = tf.convert_to_tensor(y_prediction, np.float32)
        return tf.convert_to_tensor(cluster_acc(y, y_prediction), np.float32)

    def pretrain(self, x, y, batch_size=256, epochs=200, optimizer='adam', ae_file='ae_weights.h5'):
        print('...Pretraining autoencoder...')
        #self.autoencoder.compile(loss=self.loss, optimizer=optimizer, metrics=[self.cluster_acc_2])
        #self.autoencoder.compile(loss=self.loss, optimizer=optimizer)
        es = EarlyStopping(monitor="loss", patience=50, verbose=1)
        history = self.autoencoder.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, callbacks=[es])
        """cluster_acc(y, y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, y_pred)"""
        self.autoencoder.save_weights(ae_file)
        print('Pretrained weights are saved to ./' + str(ae_file))
        self.pretrained = True
        return history

    def load_weights(self, weights_path):  # load weights of scDeepCluster model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def fit(self, x_counts, sf, y, raw_counts, batch_size=256, maxiter=2e4, tol=1e-3, update_interval=140,
            ae_weights=None, save_dir='./results/scDeepCluster', loss_weights=[1,1], optimizer='adadelta'):

        self.model.compile(loss=['kld', self.loss], loss_weights=loss_weights, optimizer=optimizer)

        print('Update interval', update_interval)
        save_interval = int(x_counts.shape[0] / batch_size) * 5  # 5 epochs
        print('Save interval', save_interval)

        # Step 1: pretrain
        if not self.pretrained and ae_weights is None:
            print('...pretraining autoencoders using default hyper-parameters:')
            print('   optimizer=\'adam\';   epochs=200')
            self.pretrain(x, batch_size)
            self.pretrained = True
        elif ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
            print('ae_weights is loaded successfully.')

        # Step 2: initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder.predict([x_counts, sf]))
        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        # logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/scDeepCluster_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()

        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict([x_counts, sf], verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(cluster_acc(y, self.y_pred), 5)
                    nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    loss = np.round(loss, 5)
                    logwriter.writerow(dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2]))
                    print('Iter-%d: ACC= %.4f, NMI= %.4f, ARI= %.4f;  L= %.5f, Lc= %.5f,  Lr= %.5f'
                          % (ite, acc, nmi, ari, loss[0], loss[1], loss[2]))

                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            if (index + 1) * batch_size > x_counts.shape[0]:
                loss = self.model.train_on_batch(x=[x_counts[index * batch_size::], sf[index * batch_size:]],
                                                 y=[p[index * batch_size::], raw_counts[index * batch_size::]])
                index = 0
            else:
                loss = self.model.train_on_batch(x=[x_counts[index * batch_size:(index + 1) * batch_size], 
                                                    sf[index * batch_size:(index + 1) * batch_size]],
                                                 y=[p[index * batch_size:(index + 1) * batch_size],
                                                    raw_counts[index * batch_size:(index + 1) * batch_size]])
                index += 1

            # save intermediate model
            if ite % save_interval == 0:
                # save scDeepCluster model checkpoints
                print('saving model to: ' + save_dir + '/scDeepCluster_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/scDeepCluster_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to: ' + save_dir + '/scDeepCluster_model_final.h5')
        self.model.save_weights(save_dir + '/scDeepCluster_model_final.h5')
        
        return self.y_pred


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
    parser.add_argument('--temp', default=500.0, type=float)
    parser.add_argument('--reducing_factor', default=0.8, type=float)
    parser.add_argument('--save_results', default=False, type=bool)

    args = parser.parse_args()

    try:
        if args.save_results:
            old_tests = pd.read_pickle("test_results.pkl")
            print("Testing... temp = {}, reducing_factor = {}".format(args.temp, args.reducing_factor))
            if(sum((old_tests['reducing_factor'] == args.reducing_factor) & (old_tests['Temperature'] == args.temp)))>0:
                print("Already done")
                exit()
            else:
                print("Factor {}".format(sum((old_tests['reducing_factor'] == args.reducing_factor) & (old_tests['Temperature'] == args.temp))))
    except:
        pass


    # load dataset
    optimizer1 = Adam(amsgrad=True, lr= 0.005)#, decay=0.05)
    optimizer2 = 'adadelta'

    data_mat = h5py.File(args.data_file)
    x = np.array(data_mat['X'])
    y = np.array(data_mat['Y'])

    # preprocessing scRNA-seq read counts matrix
    global adata
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
    scDeepCluster = SCDeepCluster(dims=[input_size, 256, 64, 32], n_clusters=args.n_clusters, noise_sd=2.5, temp=args.temp)
    plot_model(scDeepCluster.model, to_file='scDeepCluster_model.png', show_shapes=True)
    #print("autocoder summary")
    #scDeepCluster.autoencoder.summary()
    #print("model summary")
    #scDeepCluster.model.summary()
    #global xex = adata

    t0 = time()

    # Pretrain autoencoders before clustering
    #if args.ae_weights is None:
    #    scDeepCluster.pretrain(x=[adata.X, adata.obs.size_factors], y=adata.raw.X, batch_size=args.batch_size, epochs=args.pretrain_epochs, optimizer=optimizer1, ae_file=args.ae_weight_file)

    # Pretrain autoencoders before clustering
    #print(scDeepCluster.cluster_acc_2("",""))
    if args.ae_weights is None:
        interval=5
        KL = KL()

        #scDeepCluster.autoencoder.compile(loss=scDeepCluster.loss, optimizer=optimizer1)
        scDeepCluster.autoencoder.compile(loss=[scDeepCluster.loss, KL.loss], optimizer=optimizer1)
        df = pd.DataFrame(columns=['Epoch','reducing_factor','Temperature','Actual Temperature','Method','ACC','NMI','ARI','LOSS'])
        for x in range(interval, args.pretrain_epochs, interval):
            temperature = K.variable(args.temp* np.power(args.reducing_factor,x/interval),name="temperature")
            history = scDeepCluster.pretrain(x=[adata.X, adata.obs.size_factors], y=adata.raw.X, batch_size=args.batch_size, epochs=interval, optimizer=optimizer1, ae_file=args.ae_weight_file)
            #print(history.history.keys())
            #plt.plot(history.history['loss'])
            #plt.show()
            y_pred = scDeepCluster.extract_feature(x=[adata.X, adata.obs.size_factors])
            acc = np.round(cluster_acc(y, y_pred.argmax(1)), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred.argmax(1)), 5)
            ari = np.round(metrics.adjusted_rand_score(y, y_pred.argmax(1)), 5)
            loss = history.history['loss'][-1]
            if args.save_results:
                df.loc[df.shape[0]] = (x, args.reducing_factor, args.temp, args.temp* np.power(args.reducing_factor,x/interval), "argmax", acc, nmi, ari, loss)
            print('%.4f %.4f Final: ACC= %.4f, NMI= %.4f, ARI= %.4f, LOSS= %.4f, TEMP= %.4f' % (x, args.temp, acc, nmi, ari, loss,(args.temp* np.power(0.8,x/interval))))

            a = y_pred

            """pca = TSNE(n_components=2)
            x_pca = pca.fit_transform(a)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
            y_pred = kmeans.fit(x_pca).labels_
            acc = np.round(cluster_acc(y, y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
            if args.save_results:
                df.loc[df.shape[0]] = (x, args.reducing_factor, args.temp, args.temp* np.power(args.reducing_factor,x/interval), "TSNE kmeans", acc, nmi, ari, loss)
            print('TSNE + k-means result : ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))"""

            """pca = PCA(n_components=2)
            x_pca = pca.fit_transform(a)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
            y_pred = kmeans.fit(x_pca).labels_
            acc = np.round(cluster_acc(y, y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
            if args.save_results:
                df.loc[df.shape[0]] = (x, args.reducing_factor, args.temp, args.temp* np.power(args.reducing_factor,x/interval), "PCA kmeans", acc, nmi, ari, loss)
            print('PCA + k-means result : ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))"""

            #print(KMeans(n_clusters=1, n_init=20).fit(a).inertia_)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
            y_pred = kmeans.fit(a).labels_
            #print(kmeans.inertia_/KMeans(n_clusters=1, n_init=20).fit(a).inertia_, end = " ")
            #print(kmeans.cluster_centers_, end = " ")
            acc = np.round(cluster_acc(y, y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
            if args.save_results:
                df.loc[df.shape[0]] = (x, args.reducing_factor, args.temp, args.temp* np.power(args.reducing_factor,x/interval), "kmeans", acc, nmi, ari, loss)
            print('noPCA+k-means result : ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))

            print("ACC")
        
        try:
            if args.save_results:
                df = pd.concat([pd.read_pickle("test_results.pkl"), df])
        except:
            pass
        if args.save_results:
            df.to_pickle("test_results.pkl")

    
    features=scDeepCluster.extract_feature(x=[adata.X, adata.obs.size_factors])  # extract features from before clustering layer
    print(features)  # extract features from before clustering layer
    #q, _ = self.model.predict(x, verbose=0)
    #return q.argmax(1)
    print(features.shape)
    #labels=scDeepCluster.predict_clusters(x=[adata.X, adata.obs.size_factors])  # predict cluster labels using the output of clustering layer
    #print(labels)  # predict cluster labels using the output of clustering layer
    #print(np.size(features), len(labels))
    np.save("labels",features)
    """
    # begin clustering, time not include pretraining part.

    scDeepCluster.fit(x_counts=adata.X, sf=adata.obs.size_factors, y=y, raw_counts=adata.raw.X, batch_size=args.batch_size, tol=args.tol, maxiter=args.maxiter,
             update_interval=args.update_interval, ae_weights=args.ae_weights, save_dir=args.save_dir, loss_weights=[args.gamma, 1], optimizer=optimizer2)"""

    # Show the final results
    y_pred = scDeepCluster.extract_feature(x=[adata.X, adata.obs.size_factors]).argmax(1)
    print(type(y_pred))
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    #print('Final: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
    #print('Clustering time: %d seconds.' % int(time() - t0))




"""
cd /data1/antoine/scDeepCluster/code/
temp
temp
temp --n_clusters 8 --pretrain_epochs 400
temp --n_clusters 8 --pretrain_epochs 100 --ae_weights ae_weights.h5
tempCluster_model.png results/
temp
temp
temp
echo 'temp epo ACC NMI ARI LOSS' &&
for epo in `seq 10 20 150`; do
    for t in `seq 10 20 150`; do
        echo -n $t $epo " "
        python scDeepCluster.py --data_file 10X_PBMC.h5 --n_clusters 8 --pretrain_epochs $epo --temp $t 2>/dev/null |grep Final
    done #| cut -f1,2,5,7,9,11 -d" "|tr --delete ,
done



if [ -e test_results.pkl ]
then
    read -p "test_results.pkl already exists, continue ? " -n 1 -r
    #echo    # (optional) move to a new line
fi

if [[ $REPLY =~ ^[Yy]$ ]]
then
    #echo 'temp epo ACC NMI ARI LOSS'> output_results &&
    for factor in 0.3 0.5; do # 0.6 0.7 0.75 0.8 0.85 0.9 0.95 1.0; do
        for i in `seq 1 20`; do
            t=`echo "scale=10; 1.5 ^ $i" | bc`
            python scDeepCluster_tests.py --data_file 10X_PBMC.h5 --n_clusters 8 --pretrain_epochs 150 --temp $t --reducing_factor $factor --save_results True #2>/dev/null |grep Final | cut -f1,2,5,7,9,11 -d" "|tr --delete ,
        done
    done #>> output_results
fi

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('output_results', header = 0, sep =" ")
for x in sorted(set(data.values[:,1])):
    plt.plot(data.values[data.values[:,1]==x,2], label = ('temp '+ str(x)+" max = "+ str(data.values[data.values[:,1]==x,2].max())))

plt.legend()
plt.show()
"""


"""
cd /data1/antoine/scDeepCluster/code/
conda activate connect5
python
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import h5py
from sklearn import metrics
import h5py
#data_mat = h5py.File("10X_PBMC.h5")
data_mat = h5py.File("mouse_bladder_cell.h5")
x = np.array(data_mat['X'])
y_true = np.array(data_mat['Y'])
nbr_clusters_real=len(np.unique(y))

def cluster_acc(y_true, y_pred):
    """
    #Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
    #    y: true labels, numpy.array with shape `(n_samples,)`
    #    y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
    #    accuracy, in [0,1]
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

np.load('labels.npy')
a=np.load('labels.npy')
nbr_clusters_compute=a.shape[1]
plt.hist(a.max(1))
plt.show()


pca = TSNE(n_components=2)
x_pca = pca.fit_transform(a)
kmeans = KMeans(n_clusters=16, n_init=20)
y_pred = kmeans.fit(x_pca).labels_
acc = np.round(cluster_acc(y_true, y_pred), 5)
nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), 5)
print('PCA + k-means result : ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))

kmeans = KMeans(n_clusters=16, n_init=20)
y_pred = kmeans.fit(a).labels_
acc = np.round(cluster_acc(y_true, y_pred), 5)
nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), 5)
print('noPCA+k-means result : ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))

y_pred=a.argmax(1)
acc = np.round(cluster_acc(y_true, y_pred), 5)
nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), 5)
print('noPCA+nok-mea result : ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))




from collections import Counter
c = Counter( y_pred )
print( c.items() )

#plt.plot(a,y)
plt.scatter(y_pred,y)
plt.show()

plt.hist2d(y_pred,y, (nbr_clusters_compute, nbr_clusters_real), cmap=plt.cm.jet)
plt.colorbar()
plt.show()


from collections import Counter
c = Counter( y_pred )
print( c.items() )


c,_,_,_=plt.hist2d(y_pred,y, (nbr_clusters_compute, nbr_clusters_real), cmap=plt.cm.jet)
plt.close()
plt.imshow(c/c.sum(axis=0), interpolation='none')
plt.show()

"""




"""
TESTS :
- Gaussian noise negative
- 256, 64, 32 > (256, 64 || 1024, 256, 64, 32 || 512, 256, 64, 32 )
- lr= 0.005 > 0.001 > 0.015
"""
