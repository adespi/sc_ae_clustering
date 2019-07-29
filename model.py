from keras.models import Model
#from keras.losses import mse, binary_crossentropy
import keras.backend as K
#from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input, GaussianNoise, Activation, Lambda
from keras.regularizers import l2

from layers import SliceLayer, ColWiseMultLayer#, ConstantDispersionLayer
from loss import ZINB#, poisson_loss, NB

MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)

import tensorflow as tf


#START YONGJIN CODE
def build_iaf_layer(input_hidden, _name, **kwargs):
    """Inverse Autoregressive Flow
    # arguments
        input_hidden : the input of IAF layer
        _name        : must give a name
        num_trans    : number of transformations

    # options (**kwargs)
        latent_dim   : the dimension of hidden units
        pmin         : the minimum of gating functions (default: 1e-2)

    # returns
        z_layers     : latent variables
        mu_layers    : mean change variables
        sig_layers   : variance gating variables
        kl_loss      : KL divergence loss
    """

    _stoch = build_gaussian_stoch_layer(input_hidden, name=_name, **kwargs)
    z, mu0, sig0, kl_loss = _stoch
            
    z_layers = [z]
    mu_layers = [mu0]
    sig_layers = [sig0]
            
    num_trans = kwargs.get('num_trans', 1)
            
    for l in range(1, num_trans + 1):
        hdr = '%s_%d'%(_name, l)
        z, mu, sig, _kl = add_iaf_transformation(input_hidden, z, name=hdr, **kwargs)
        z_layers.append(z)
        mu_layers.append(mu)
        sig_layers.append(sig)
        kl_loss -= _kl
        
    return z_layers, mu_layers, sig_layers, kl_loss


def _sigmoid(x, pmin=1e-4, pmax=1.0 - 1e-4):
    return K.sigmoid(x) * (pmax - pmin) + pmin


def build_gaussian_stoch_layer(hh, **kwargs):
    """VAE Gaussian layer

    # arguments
        hh         : the input of this layer

    # options (**kwargs)
        latent_dim : the dimension of hidden units

    # returns
        z_stoch    : latent variables
        mu         : mean change variables
        sig        : variance gating variables
        kl_loss    : KL divergence loss

    """
            
    d = K.int_shape(hh)[1]
    latent_dim = kwargs.get('latent_dim', d)

    _name_it = keras_name_func(**kwargs)
            
    def _sample(args):
        mu, sig = args
        batch = K.shape(mu)[0]
        dim = K.int_shape(mu)[1]
        epsilon = K.random_normal(shape=(batch, dim), mean=0.0, stddev=1.0)
        return mu + sig * epsilon

    mu = Dense(latent_dim, activation='linear', name=_name_it('mu'))(hh)
    sig = Dense(latent_dim, activation=_sigmoid, name=_name_it('sig'))(hh)
    z_stoch = Lambda(_sample, output_shape=(latent_dim,), name=_name_it('stoch'))([mu,sig])
            
    kl_loss = 1.0 + 2.0 * K.log(sig) - K.square(mu) - K.square(sig)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
            
    return z_stoch, mu, sig, kl_loss


def add_iaf_transformation(hh, z_prev, **kwargs):
    """
    """

    rank = kwargs.get('rank',None)
    act = kwargs.get('act','linear')
    pmin = kwargs.get('pmin',1e-2)
    l2_penalty = kwargs.get('l2_penalty',0.01)

    _name_it = keras_name_func(**kwargs)

    d = K.int_shape(z_prev)[1]
    mu = Dense(d, activation=act, activity_regularizer=l2(l2_penalty), name=_name_it('mu'))(hh)
    _transform = lambda msz: msz[0] * (1.0 - msz[1]) + msz[2] * msz[1]

    if rank is not None:
        # Introduce bottleneck
        hh_bn = Dense(rank, activation=act, name=_name_it('bn'))(hh)
        sig = Dense(d, activation=_sigmoid, name=_name_it('sig'))(hh_bn)
    else:
        sig = Dense(d, activation=_sigmoid, name=_name_it('sig'))(hh)

    z_next = Lambda(_transform, output_shape=(d,), name=_name_it('z'))([mu,sig,z_prev])
    kl_loss = K.sum(K.log(sig), axis=-1)
    return z_next, mu, sig, kl_loss

def keras_name_func(**kwargs):
    _name = kwargs.get('name',None)
    _name_0 = lambda x: None
    _name_1 = lambda x: '%s_%s'%(_name, str(x))
    return _name_1 if _name is not None else _name_0


#END YONGJIN CODE



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



def create_model(model, dims, act = 'relu', init='glorot_uniform', ridge=0, debug=False, **kwargs):
    #n_clusters=args.n_clusters
    ##alpha=1.0,
    assert model in ["ae", "vae", "iaf"]
    #if model == "ae":
    noise_sd = kwargs.get('noise_sd', 2.5)
    if model == "iaf":
        num_trans = kwargs.get('num_trans', 6)
    
    n_stacks = len(dims) - 1

    # input
    counts_input = Input(shape=(dims[0],), name='counts')
    h = counts_input
    #if model == "ae":
    h = GaussianNoise(noise_sd, name='input_noise')(h)
    
    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], kernel_initializer=init, name='encoder_%d' % i)(h)
        if model == "ae" or model == "vae" :
            h = GaussianNoise(noise_sd, name='noise_%d' % i)(h)    # add Gaussian noise
        h = Activation(act)(h)

    # hidden layer
    if model == "ae":
        h = Dense(dims[-1], kernel_initializer=init, name='encoder_hidden')(h)  # hidden layer, features are extracted from here
        latent_layer = h
    elif model == "vae":
        z_mean = Dense(dims[-1], name='z_mean')(h)
        z_log_var = Dense(dims[-1], name='z_log_var')(h)
        z = Lambda(sampling, output_shape=(dims[-1],), name='z')([z_mean, z_log_var])
        h = z
        latent_layer = z_mean
    else :
        z, mu, sig, kl = build_iaf_layer(h, _name='IAF', num_trans = num_trans, latent_dim = dims[-1])
        z = z[-1]
        mu = mu[-1]
        h = z
        latent_layer = mu

    sf_layer = Input(shape=(1,), name='size_factors')

    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        h = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(h)

    # output
    
    pi = Dense(dims[0], activation='sigmoid', kernel_initializer=init, name='pi')(h)

    disp = Dense(dims[0], activation=DispAct, kernel_initializer=init, name='dispersion')(h)

    mean = Dense(dims[0], activation=MeanAct, kernel_initializer=init, name='mean')(h)

    adjusted_mean = ColWiseMultLayer(name='output')([mean, sf_layer])
    outputs = SliceLayer(0, name='slice')([adjusted_mean, disp, pi])

    model_network = Model([counts_input, sf_layer], outputs, name=model+'_mlp')
    encoder_network = Model(counts_input, latent_layer, name='encoder')
    imputation_no_zi_network = Model([counts_input, sf_layer], adjusted_mean, name=model+'_mlp')

    # loss
    zinb = ZINB(pi, theta=disp, ridge_lambda=ridge, debug=debug)
    if model == "ae":
        def loss(y_true, y_pred):
            return zinb.loss(y_true= y_true, y_pred= y_pred)
    elif model == "vae":
        def loss(y_true, y_pred):
            reconstruction_loss = zinb.loss(y_true= y_true, y_pred= y_pred)# tf.get_variable(output, (17925, 73909)))
            reconstruction_loss *= dims[0]
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            #kl_loss = K.mean(kl_loss, axis=-1)
            kl_loss *= -0.5
            vae_loss = K.mean(reconstruction_loss + kl_loss)
            return vae_loss
            #return reconstruction_loss
    else :
        def loss(y_true, y_pred):
            reconstruction_loss = zinb.loss(y_true= y_true, y_pred= y_pred)# tf.get_variable(output, (17925, 73909)))
            reconstruction_loss *= dims[0]
            vae_loss = K.mean(reconstruction_loss + kl)
            return vae_loss
            #return reconstruction_loss
    return model_network, encoder_network, imputation_no_zi_network, loss, counts_input, latent_layer
