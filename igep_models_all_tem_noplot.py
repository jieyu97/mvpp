"""
@author: Tim Janke, Energy Information Networks & Systems Lab @ TU Darmstadt, Germany

Module for IGEP class.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.losses import Loss
# import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# define energy score
def energy_score(y_true, S):
    """
    Computes energy score:

    Parameters
    ----------
    y_true : tf tensor of shape (BATCH_SIZE, D, 1)
        True values.
    S : tf tensor of shape (BATCH_SIZE, D, N_SAMPLES)
        Predictive samples.

    Returns
    -------
    tf tensor of shape (BATCH_SIZE,)
        Scores.

    """
    # y_true = tf.cast(y_true, tf.float32)
    # S = tf.cast(S, tf.float32)
    
    beta=1
    n_samples = S.shape[-1]
    def expected_dist(diff, beta):
        return K.sum(K.pow(K.sqrt(K.sum(K.square(diff), axis=-2)+K.epsilon()), beta),axis=-1)
    es_1 = expected_dist(y_true - S, beta)
    es_2 = 0
    for i in range(n_samples):
        es_2 = es_2 + expected_dist(K.expand_dims(S[:,:,i]) - S, beta)
        
    # es_1 = tf.cast(es_1, tf.float32)
    # es_2 = tf.cast(es_2, tf.float32)
    n_samples = tf.cast(n_samples, tf.float32)
    es = es_1/n_samples - es_2/(2*n_samples**2)
    es = tf.cast(es, tf.float32)
    return es


# subclass tensorflow.keras.losses.Loss
class EnergyScore(Loss):
    def call(self, y_true, S):
        return energy_score(y_true, S)


class igep(object):
    """
    Class for Implicit Generative Ensemble Postprocessing (IGEP) models.
    IGEP models can be used to generate samples from an implicit multivariate predictive distribution.
    See Janke&Steinke (2020): "Probabilistic multivariate electricity price forecasting using implicit generative ensemble post-processing"
    https://arxiv.org/abs/2005.13417
    
    Passing model_type = 1 will create original model.
    Passing model_type = 2 will create new advanced model.

    Parameters
    ----------
    dim_out : int
        Number of output dimensions.
    dim_in_mean : int
        Number of features used for predictive mean.
    dim_in_noise : int
        Number of features used for uncertainty. Will be ignored if model_type=1.
    dim_latent : int
        Number of latent variables.
    n_samples_train : int
        Number of predictive samples to be drawn in training.
        More samples should results in improved accuracy but takes longer to train.
    model_type : int, 1 or 2
            1 will create original model, 2 will create an improved and more flexible model.
    latent_dist : sting, optional
        Family of the latent distributions. Options are uniform and normal. The default is "uniform".
    latent_dist_params : tuple, optional
        Parameters for latent distributions. (min,max) for uniform, (mean,stddev) for normal. 
        If None is passed params are set to (-1,1) and (0,1) respectively.

    Returns
    -------
    None.

    """
    
    def __init__(self, dim_out, dim_in_mean, dim_in_std, dim_in_features, dim_latent, n_samples_train, 
                 layer_number, nodes_number, model_type, latent_dist="uniform", latent_dist_params=None):
        
        self.dim_out = dim_out
        self.dim_in_mean = dim_in_mean
        self.dim_in_std = dim_in_std
        self.dim_in_features = dim_in_features
        self.dim_latent = dim_latent
        self.n_samples_train = n_samples_train
        self.latent_dist = latent_dist
        if latent_dist_params is None:
            if self.latent_dist == "uniform":
                self.latent_dist_params = (-1.0, 1.0)
            elif self.latent_dist == "normal":
                self.latent_dist_params = (0.0, 1.0)
        else:
            self.latent_dist_params = latent_dist_params
        
        self._n_samples = n_samples_train
        
        if model_type == 1:
            self._build_model = self._build_model_1
        elif model_type == 21:
            self._build_model = self._build_model_21
        elif model_type == 22:
            self._build_model = self._build_model_22
        elif model_type == 231:
            self._build_model = self._build_model_231
        elif model_type == 232:
            self._build_model = self._build_model_232
        elif model_type == 31:
            self._build_model = self._build_model_31
        elif model_type == 321:
            self._build_model = self._build_model_321
        elif model_type == 322:
            self._build_model = self._build_model_322
        elif model_type == 323:
            self._build_model = self._build_model_323
        elif model_type == 325:
            self._build_model = self._build_model_325
        elif model_type == 326:
            self._build_model = self._build_model_326
            
        self.layer_number = layer_number
        self.nodes_number = nodes_number
        
        self.model = self._build_model()
    
    
    
    def _build_model_1(self):
        """
        Defines original IGEP model:
        Variance of dim_out of the latent distributions depend on the ensemble spread.
        See Janke&Steinke (2020): "Probabilistic multivariate electricity price forecasting using implicit generative ensemble post-processing"
        https://arxiv.org/abs/2005.13417

        Returns
        -------
        object
            Keras model.

        """
        
        ### Inputs ###
        x_mean = keras.Input(shape=(self.dim_out, self.dim_in_mean), name = "x_mean")
        delta = keras.Input(shape=(self.dim_out,1), name = "delta")
        bs = K.shape(delta)[0]

        
        ### mean model ###
        mu =  layers.LocallyConnected1D(filters=1, 
                                        kernel_size=1, 
                                        strides=1,
                                        padding='valid',
                                        data_format='channels_last',
                                        use_bias=True,
                                        activation='linear')(x_mean) # [n_dim_out x 1] 
        mu = layers.Lambda(lambda arg: K.repeat_elements(arg, self._n_samples, axis=-1))(mu) # [n_dim_out x n_samples]     
        
        #### noise model ###
        # generate noise
        if self.latent_dist == "uniform":
            u = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                            minval=-1.0, 
                                                            maxval=1.0))([bs, self.dim_out, self._n_samples])
            v = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                            minval=self.latent_dist_params[0], 
                                                            maxval=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])

        elif self.latent_dist == "normal":
            u = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                            mean=0.0, 
                                                            stddev=1.0))([bs, self.dim_out, self._n_samples])
            v = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                            mean=self.latent_dist_params[0], 
                                                            stddev=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
        
        u = layers.Multiply()([delta, u]) # adapt u samples by ensemble spread
        
        # decode samples from adaptive latent variables
        # ("channels_first" produces an error, therefore we use channels_last + 2 x permute_dims)
        u = layers.Lambda(lambda arg: K.permute_dimensions(arg, (0,2,1)))(u)
        eps_u = layers.Conv1D(filters=self.dim_out, 
                              kernel_size=1,
                              strides=1,
                              padding="valid",
                              data_format="channels_last",
                              activation="linear", 
                              use_bias=False)(u)
        eps_u = layers.Lambda(lambda arg: K.permute_dimensions(arg, (0,2,1)))(eps_u)
        
        # decode samples from independent latent variables
        v = layers.Lambda(lambda arg: K.permute_dimensions(arg, (0,2,1)))(v)
        eps_v = layers.Conv1D(filters=self.dim_out, 
                              kernel_size=1,
                              strides=1,
                              padding="valid",
                              data_format="channels_last",
                              activation="linear", 
                            use_bias=False)(v)
        eps_v = layers.Lambda(lambda arg: K.permute_dimensions(arg, (0,2,1)))(eps_v)
        
        #### add noise to mean ###
        y = layers.Add()([mu, eps_u, eps_v])
        
        return Model(inputs=[x_mean, delta], outputs=y)



    def _build_model_21(self):
        """
        Defines igep model 2.0.
        In this model, all latent varaibles are independet, but for each output 
        dimensions we learn a set of weights to scale the samples from the latent variables.
        The model for the weights can possibly depend on a large set of arbitrary inputs.

        Returns
        -------
        object
            Keras model.

        """
        
        ### Inputs ###
        x_mean = keras.Input(shape=(self.dim_out, self.dim_in_mean), name = "x_mean")
        x_noise = keras.Input(shape=(self.dim_out, self.dim_in_features), name = "x_noise")
        bs = K.shape(x_noise)[0]
        
        ### mean model ###
        mu =  layers.LocallyConnected1D(filters=1, 
                                        kernel_size=1, 
                                        strides=1,
                                        padding='valid',
                                        data_format='channels_last',
                                        use_bias=True,
                                        activation='linear')(x_mean) # [?,n_dim_out x,1] 
        mu = layers.Lambda(lambda arg: K.repeat_elements(arg, self._n_samples, axis=-1))(mu) # [?, n_dim_out, n_samples]     
        

        #### noise model ###
        W = layers.LocallyConnected1D(filters=self.dim_latent, 
                                      kernel_size=1, 
                                      strides=1,
                                      padding='valid',
                                      data_format='channels_last',
                                      use_bias=True,
                                      activation='linear')(x_noise) # (?, dim_out, dim_latent)
        
        if self.latent_dist == "uniform":
            z = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                     minval=self.latent_dist_params[0], 
                                                     maxval=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
        elif self.latent_dist == "normal":
            z = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                    mean=self.latent_dist_params[0], 
                                                    stddev=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])

        #### add noise to mean ###
        eps = layers.Dot(axes=(2,1))([W,z]) # [?,n_dim_out,n_samples]
        y = layers.Add()([mu,eps])
        
        return Model(inputs=[x_mean, x_noise], outputs=y)



    def _build_model_22(self):
        """
        Defines igep model 2.0.
        In this model, all latent varaibles are independet, but for each output 
        dimensions we learn a set of weights to scale the samples from the latent variables.
        The model for the weights can possibly depend on a large set of arbitrary inputs.

        Returns
        -------
        object
            Keras model.

        """
        
        ### Inputs ###
        x_mean = keras.Input(shape=(self.dim_out, self.dim_in_mean), name = "x_mean")
        x_noise = keras.Input(shape=(self.dim_out, self.dim_in_features), name = "x_noise")
        bs = K.shape(x_noise)[0]
        
        ### mean model ###
        mu =  layers.LocallyConnected1D(filters=1, 
                                        kernel_size=1, 
                                        strides=1,
                                        padding='valid',
                                        data_format='channels_last',
                                        use_bias=True,
                                        activation='linear')(x_mean) # [?,n_dim_out x,1] 
        mu = layers.Lambda(lambda arg: K.repeat_elements(arg, self._n_samples, axis=-1))(mu) # [?, n_dim_out, n_samples]     
        

        #### noise model ###
        ### If each element in W should depend on complete input: ###
        W = layers.Flatten()(x_noise)
        for l in range(self.layer_number):
            W = layers.Dense(self.nodes_number, use_bias=True, activation = 'elu')(W)
        
        W = layers.Dense(self.dim_out*self.dim_latent, use_bias=True, activation = 'linear')(W) # (?, dim_out*dim_latent)
        W = layers.Reshape((self.dim_out, self.dim_latent))(W) # (?, dim_out, dim_latent)
        ##################################################################
        
        if self.latent_dist == "uniform":
            z = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                     minval=self.latent_dist_params[0], 
                                                     maxval=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
        elif self.latent_dist == "normal":
            z = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                    mean=self.latent_dist_params[0], 
                                                    stddev=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])

        #### add noise to mean ###
        eps = layers.Dot(axes=(2,1))([W,z]) # [?,n_dim_out,n_samples]
        y = layers.Add()([mu,eps])
        
        return Model(inputs=[x_mean, x_noise], outputs=y)
    
    
    
    def _build_model_231(self):
        """
        Defines igep model 3.0.
        In this model, all latent varaibles are independet, but for each output 
        dimensions we learn a set of weights to scale the samples from the latent variables.
        The model for the weights can possibly depend on a large set of arbitrary inputs.

        Returns
        -------
        object
            Keras model.

        """
        
        ### Inputs ###
        x_mean = keras.Input(shape=(self.dim_out, self.dim_in_mean), name = "x_mean")
        aux_mu = keras.Input(shape=(self.dim_out, (self.dim_in_features-1) ), name = "aux_mu")
        x_aux_sigma = keras.Input(shape=(self.dim_out, self.dim_in_features), name = "x_aux_sigma")
        bs = K.shape(x_aux_sigma)[0]
        
        ### mean model ###
        mu =  layers.LocallyConnected1D(filters=1, 
                                        kernel_size=1, 
                                        strides=1,
                                        padding='valid',
                                        data_format='channels_last',
                                        use_bias=True,
                                        activation='linear')(x_mean) # [?,n_dim_out x,1] 
        mu = layers.Lambda(lambda arg: K.repeat_elements(arg, self._n_samples, axis=-1))(mu) # [?, n_dim_out, n_samples]     
        

        #### noise model ###
        # W for mean
        W_mu = layers.Flatten()(aux_mu)
        for l in range(self.layer_number):
            W_mu = layers.Dense(self.nodes_number, use_bias=True, activation = 'elu')(W_mu)
        
        W_mu = layers.Dense(self.dim_out*self.dim_latent, use_bias=True, activation = 'linear')(W_mu) # (?, dim_out*dim_latent)
        W_mu = layers.Reshape((self.dim_out, self.dim_latent))(W_mu) # (?, dim_out, dim_latent)
        # W for sd
        W_sigma = layers.Flatten()(x_aux_sigma)
        for l in range(self.layer_number):
            W_sigma = layers.Dense(self.nodes_number, use_bias=True, activation = 'elu')(W_sigma)
        
        W_sigma = layers.Dense(self.dim_out*self.dim_latent, use_bias=True, activation = 'linear')(W_sigma) # (?, dim_out*dim_latent)
        W_sigma = layers.Reshape((self.dim_out, self.dim_latent))(W_sigma) # (?, dim_out, dim_latent)
        ############################################################
        
        if self.latent_dist == "uniform":
            z = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                     minval=self.latent_dist_params[0], 
                                                     maxval=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
        elif self.latent_dist == "normal":
            z = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                    mean=self.latent_dist_params[0], 
                                                    stddev=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])

        #### add noise to mean ###
        W = layers.Add()([W_mu,W_sigma])
        eps = layers.Dot(axes=(2,1))([W,z]) # [?,n_dim_out,n_samples]
        y = layers.Add()([mu,eps])
        
        return Model(inputs=[x_mean, aux_mu, x_aux_sigma], outputs=y)
    
    
    
    def _build_model_232(self):
        """
        Defines igep model 3.0.
        In this model, all latent varaibles are independet, but for each output 
        dimensions we learn a set of weights to scale the samples from the latent variables.
        The model for the weights can possibly depend on a large set of arbitrary inputs.

        Returns
        -------
        object
            Keras model.

        """
        
        ### Inputs ###
        x_mean = keras.Input(shape=(self.dim_out, self.dim_in_mean), name = "x_mean")
        aux_mu = keras.Input(shape=(self.dim_out, (self.dim_in_features-1) ), name = "aux_mu")
        x_aux_sigma = keras.Input(shape=(self.dim_out, self.dim_in_features), name = "x_aux_sigma")
        bs = K.shape(x_aux_sigma)[0]
        
        ### mean model ###
        mu =  layers.LocallyConnected1D(filters=1, 
                                        kernel_size=1, 
                                        strides=1,
                                        padding='valid',
                                        data_format='channels_last',
                                        use_bias=True,
                                        activation='linear')(x_mean) # [?,n_dim_out x,1] 
        mu = layers.Lambda(lambda arg: K.repeat_elements(arg, self._n_samples, axis=-1))(mu) # [?, n_dim_out, n_samples]     
        

        #### noise model ###
        # W for mean
        W_mu = layers.Flatten()(aux_mu)
        for l in range(self.layer_number):
            W_mu = layers.Dense(self.nodes_number, use_bias=True, activation = 'elu')(W_mu)
        
        W_mu = layers.Dense(self.dim_out*self.dim_latent, use_bias=True, activation = 'linear')(W_mu) # (?, dim_out*dim_latent)
        W_mu = layers.Reshape((self.dim_out, self.dim_latent))(W_mu) # (?, dim_out, dim_latent)
        # W for sd
        W_sigma = layers.Flatten()(x_aux_sigma)
        for l in range(self.layer_number):
            W_sigma = layers.Dense(self.nodes_number, use_bias=True, activation = 'elu')(W_sigma)
        
        W_sigma = layers.Dense(self.dim_out*self.dim_latent, use_bias=True, activation = 'linear')(W_sigma) # (?, dim_out*dim_latent)
        W_sigma = layers.Reshape((self.dim_out, self.dim_latent))(W_sigma) # (?, dim_out, dim_latent)
        ############################################################
        
        if self.latent_dist == "uniform":
            z_mu = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                     minval=self.latent_dist_params[0], 
                                                     maxval=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
        elif self.latent_dist == "normal":
            z_mu = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                    mean=self.latent_dist_params[0], 
                                                    stddev=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])

        if self.latent_dist == "uniform":
            z_sigma = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                     minval=self.latent_dist_params[0], 
                                                     maxval=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
        elif self.latent_dist == "normal":
            z_sigma = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                    mean=self.latent_dist_params[0], 
                                                    stddev=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
        
        #### add noise to mean ###
        eps_mu = layers.Dot(axes=(2,1))([W_mu,z_mu]) # [?,n_dim_out,n_samples]
        eps_sigma = layers.Dot(axes=(2,1))([W_sigma,z_sigma]) # [?,n_dim_out,n_samples]
        y = layers.Add()([mu,eps_mu,eps_sigma])
        
        return Model(inputs=[x_mean, aux_mu, x_aux_sigma], outputs=y)
    
    

    def _build_model_31(self):
        """
        Extended IGEP model with conditional noise;

        Returns
        -------
        object
            Keras model.

        """
        
        ### Inputs ###
        all_input = keras.Input(shape=(self.dim_out, self.dim_in_features), name = "all_input")
        bs = K.shape(all_input)[0]
        
        n1 = layers.LocallyConnected1D(filters=2, 
                                       kernel_size=1, 
                                       strides=1,
                                       padding='valid',
                                       data_format='channels_last',
                                       use_bias=True,
                                       activation='linear')(all_input) # (?, dim_out, 2) 
        
        n2 = layers.Flatten()(n1)
        
        n21 = layers.Dense(self.dim_latent, activation = 'linear')(n2) # means of latent variables
        n22 = layers.Dense(self.dim_latent, activation = 'exponential')(n2) # spread of latent variables 
        # activations.exponential
        
        if self.latent_dist == "uniform":
            z = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                     minval=self.latent_dist_params[0], 
                                                     maxval=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
        elif self.latent_dist == "normal":
            z = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                    mean=self.latent_dist_params[0], 
                                                    stddev=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])

        n22_reshape = layers.Lambda(lambda arg: K.reshape(arg, (bs, self.dim_latent, 1)))(n22) # [dim_latent x 1]
        
        z_spread = layers.Multiply()([n22_reshape, z]) # [dim_latent x n_samples] 
        
        # z_spread_permute = layers.Lambda(lambda arg: K.permute_dimensions(arg, (0,2,1)))(z_spread) # [dim_latent x n_samples] 
        
        n21_reshape = layers.Lambda(lambda arg: K.reshape(arg, (bs, self.dim_latent, 1)))(n21) # [dim_latent x 1]
        
        n21_repeat = layers.Lambda(lambda arg: K.repeat_elements(arg, self._n_samples, axis=-1))(n21_reshape) # [dim_latent x n_samples]  
        
        z_location = layers.Add()([n21_repeat, z_spread]) # [dim_latent x n_samples] 
        
        z_samples = layers.Lambda(lambda arg: K.permute_dimensions(arg, (0,2,1)))(z_location) # (, n_samples, dim_latent)
        
        g = layers.Conv1D(filters=self.dim_out, 
                          kernel_size=1,
                          strides=1,
                          padding="valid",
                          data_format="channels_last",
                          activation="linear", 
                          use_bias=False)(z_samples) # (bs, n_samples, dim_out)
        
        y = layers.Lambda(lambda arg: K.permute_dimensions(arg, (0,2,1)))(g) # (, dim_out, n_samples)
        
        return Model(inputs=[all_input], outputs=y)

        
        
    def _build_model_321(self):        

        ### Inputs ###
        input_mean = keras.Input(shape=(self.dim_out, self.dim_in_features), name = "input_mean")
        input_sd = keras.Input(shape=(self.dim_out, self.dim_in_features), name = "input_sd")
        bs = K.shape(input_mean)[0]
        
        
        x_mean = layers.LocallyConnected1D(filters=1, 
                                       kernel_size=1, 
                                       strides=1,
                                       padding='valid',
                                       data_format='channels_last',
                                       use_bias=True,
                                       activation='linear')(input_mean) # (, dim_out, 1)
        
        x_mean_all = layers.Lambda(lambda arg: K.repeat_elements(arg, self._n_samples, axis=-1))(x_mean) # (, dim_out, n_samples)
        
        z_delta = layers.LocallyConnected1D(filters=1, 
                                       kernel_size=1, 
                                       strides=1,
                                       padding='valid',
                                       data_format='channels_last',
                                       use_bias=True,
                                       activation='linear')(input_sd) # (, dim_out, 1)
        
        z_delta_flat = layers.Flatten()(z_delta)
        
        z_delta_final = layers.Dense(self.dim_latent, activation = 'exponential')(z_delta_flat) # spread of latent variables 
        
        if self.latent_dist == "uniform":
            z = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                     minval=self.latent_dist_params[0], 
                                                     maxval=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
        elif self.latent_dist == "normal":
            z = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                    mean=self.latent_dist_params[0], 
                                                    stddev=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
       
        z_delta_reshape = layers.Lambda(lambda arg: K.reshape(arg, (bs, self.dim_latent, 1)))(z_delta_final) # (, dim_latent, 1)
        
        z_adjust_spread = layers.Multiply()([z_delta_reshape, z]) # (, dim_latent, n_samples)
        
        z_samples = layers.Lambda(lambda arg: K.permute_dimensions(arg, (0,2,1)))(z_adjust_spread) # (, n_samples, dim_latent)
        
        g = layers.Conv1D(filters=self.dim_out, 
                          kernel_size=1,
                          strides=1,
                          padding="valid",
                          data_format="channels_last",
                          activation="linear", 
                          use_bias=False)(z_samples) # (, n_samples, dim_out)
        
        g_permute = layers.Lambda(lambda arg: K.permute_dimensions(arg, (0,2,1)))(g) # (, dim_out, n_samples)
        
        y = layers.Add()([x_mean_all, g_permute]) # (, dim_out, n_samples)


        return Model(inputs=[input_mean, input_sd], outputs=y)
    
    

    def _build_model_322(self):        

        ### Inputs ###
        input_mean = keras.Input(shape=(self.dim_out, self.dim_in_mean), name = "input_mean")
        input_all = keras.Input(shape=(self.dim_out, self.dim_in_features), name = "input_all")
        bs = K.shape(input_mean)[0]
        
        
        x_mean = layers.LocallyConnected1D(filters=1, 
                                       kernel_size=1, 
                                       strides=1,
                                       padding='valid',
                                       data_format='channels_last',
                                       use_bias=True,
                                       activation='linear')(input_mean) # (, dim_out, 1)
        
        x_mean_all = layers.Lambda(lambda arg: K.repeat_elements(arg, self._n_samples, axis=-1))(x_mean) # (, dim_out, n_samples)
        
        
        if self.latent_dist == "uniform":
            z = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                     minval=self.latent_dist_params[0], 
                                                     maxval=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
        elif self.latent_dist == "normal":
            z = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                    mean=self.latent_dist_params[0], 
                                                    stddev=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
       
        # weights
        W = layers.Flatten()(input_all)
        for l in range(self.layer_number):
            W = layers.Dense(self.nodes_number, use_bias=True, activation = 'elu')(W)
        
        W = layers.Dense(self.dim_out*self.dim_latent, use_bias=True, activation = 'linear')(W) # (, dim_out*dim_latent)
        W = layers.Reshape((self.dim_out, self.dim_latent))(W) # (, dim_out, dim_latent)
        ##################################################################
        
        z_samples = layers.Dot(axes=(2,1))([W,z]) # (, dim_out, n_samples)
        y = layers.Add()([x_mean_all,z_samples])


        return Model(inputs=[input_mean, input_all], outputs=y)


    
    def _build_model_323(self):        

        ### Inputs ###
        input_mean = keras.Input(shape=(self.dim_out, self.dim_in_mean), name = "input_mean")
        input_sd = keras.Input(shape=(self.dim_out, self.dim_in_std), name = "input_sd")
        input_all = keras.Input(shape=(self.dim_out, self.dim_in_features), name = "input_all")
        bs = K.shape(input_mean)[0]
        
        
        x_mean = layers.LocallyConnected1D(filters=1, 
                                       kernel_size=1, 
                                       strides=1,
                                       padding='valid',
                                       data_format='channels_last',
                                       use_bias=True,
                                       activation='linear')(input_mean) # (, dim_out, 1)
        
        x_mean_all = layers.Lambda(lambda arg: K.repeat_elements(arg, self._n_samples, axis=-1))(x_mean) # (, dim_out, n_samples)
        
        z_delta = layers.LocallyConnected1D(filters=1, 
                                       kernel_size=1, 
                                       strides=1,
                                       padding='valid',
                                       data_format='channels_last',
                                       use_bias=True,
                                       activation='linear')(input_sd) # (, dim_out, 1)
        
        z_delta_flat = layers.Flatten()(z_delta)
        
        z_delta_final = layers.Dense(self.dim_latent, activation = 'exponential')(z_delta_flat) # spread of latent variables 
        
        z_delta_reshape = layers.Lambda(lambda arg: K.reshape(arg, (bs, self.dim_latent, 1)))(z_delta_final) # (, dim_latent, 1)
        
        if self.latent_dist == "uniform":
            z = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                     minval=self.latent_dist_params[0], 
                                                     maxval=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
        elif self.latent_dist == "normal":
            z = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                    mean=self.latent_dist_params[0], 
                                                    stddev=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
       
        z_adjust_spread = layers.Multiply()([z_delta_reshape, z]) # (, dim_latent, n_samples)
        
        # weights
        W = layers.Flatten()(input_all)
        for l in range(self.layer_number):
            W = layers.Dense(self.nodes_number, use_bias=True, activation = 'elu')(W)
        
        W = layers.Dense(self.dim_out*self.dim_latent, use_bias=True, activation = 'linear')(W) # (, dim_out*dim_latent)
        W = layers.Reshape((self.dim_out, self.dim_latent))(W) # (, dim_out, dim_latent)
        ##################################################################
        
        z_samples = layers.Dot(axes=(2,1))([W,z_adjust_spread]) # (, dim_out, n_samples)
        y = layers.Add()([x_mean_all,z_samples])


        return Model(inputs=[input_mean, input_sd, input_all], outputs=y)


    
    def _build_model_325(self):        

        ### Inputs ###
        input_mean = keras.Input(shape=(self.dim_out, self.dim_in_mean), name = "input_mean")
        input_sd = keras.Input(shape=(self.dim_out, self.dim_in_std), name = "input_sd")
        input_all = keras.Input(shape=(self.dim_out, self.dim_in_features), name = "input_all")
        bs = K.shape(input_mean)[0]
        
        #####
        x_mean = layers.Flatten()(input_mean)
        
        x_mean = layers.Dense(25, use_bias=True, activation = 'elu')(x_mean)
        x_mean = layers.Dense(25, use_bias=True, activation = 'elu')(x_mean)
        
        x_mean = layers.Dense(self.dim_out, use_bias=True, activation = 'linear')(x_mean) # (, dim_out*1)
        x_mean = layers.Reshape((self.dim_out, 1))(x_mean) # (, dim_out, 1)
        #####
        
        x_mean_all = layers.Lambda(lambda arg: K.repeat_elements(arg, self._n_samples, axis=-1))(x_mean) # (, dim_out, n_samples)
        
        #####
        z_delta = layers.Flatten()(input_sd)
        
        z_delta = layers.Dense(25, use_bias=True, activation = 'elu')(z_delta)
        z_delta = layers.Dense(25, use_bias=True, activation = 'elu')(z_delta)
        
        z_delta = layers.Dense(self.dim_out, use_bias=True, activation = 'linear')(z_delta) # (, dim_out*1)
        #####
        
        z_delta_flat = layers.Flatten()(z_delta)
        
        z_delta_final = layers.Dense(self.dim_latent, activation = 'exponential')(z_delta_flat) # spread of latent variables 
        
        z_delta_reshape = layers.Lambda(lambda arg: K.reshape(arg, (bs, self.dim_latent, 1)))(z_delta_final) # (, dim_latent, 1)
        
        if self.latent_dist == "uniform":
            z = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                     minval=self.latent_dist_params[0], 
                                                     maxval=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
        elif self.latent_dist == "normal":
            z = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                    mean=self.latent_dist_params[0], 
                                                    stddev=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
       
        z_adjust_spread = layers.Multiply()([z_delta_reshape, z]) # (, dim_latent, n_samples)
        
        # weights
        #####
        W = layers.Flatten()(input_all)
        for l in range(self.layer_number):
            W = layers.Dense(self.nodes_number, use_bias=True, activation = 'elu')(W)
        
        W = layers.Dense(self.dim_out*self.dim_latent, use_bias=True, activation = 'linear')(W) # (, dim_out*dim_latent)
        W = layers.Reshape((self.dim_out, self.dim_latent))(W) # (, dim_out, dim_latent)
        #####
        
        z_samples = layers.Dot(axes=(2,1))([W, z_adjust_spread]) # (, dim_out, n_samples)
        y = layers.Add()([x_mean_all, z_samples])


        return Model(inputs=[input_mean, input_sd, input_all], outputs=y)
    
    

    def _build_model_326(self):        

        ### Inputs ###
        input_mean = keras.Input(shape=(self.dim_out, self.dim_in_mean), name = "input_mean")
        input_sd = keras.Input(shape=(self.dim_out, self.dim_in_std), name = "input_sd")
        input_all = keras.Input(shape=(self.dim_out, self.dim_in_features), name = "input_all")
        bs = K.shape(input_mean)[0]
        
        #####
        x_mean = layers.LocallyConnected1D(filters=1, 
                                       kernel_size=1, 
                                       strides=1,
                                       padding='valid',
                                       data_format='channels_last',
                                       use_bias=True,
                                       activation='linear')(input_mean) # (, dim_out, 1)
        
        x_mean_all = layers.Lambda(lambda arg: K.repeat_elements(arg, self._n_samples, axis=-1))(x_mean) # (, dim_out, n_samples)
        
        #####
        z_delta = layers.LocallyConnected1D(filters=20, 
                                       kernel_size=1, 
                                       strides=1,
                                       padding='valid',
                                       data_format='channels_last',
                                       use_bias=True,
                                       activation='linear')(input_sd) # (, dim_out, 8)
        z_delta = layers.LocallyConnected1D(filters=1, 
                                       kernel_size=1, 
                                       strides=1,
                                       padding='valid',
                                       data_format='channels_last',
                                       use_bias=True,
                                       activation='linear')(z_delta) # (, dim_out, 1)
        
        z_delta_flat = layers.Flatten()(z_delta)
        
        z_delta_final = layers.Dense(self.dim_latent, activation = 'exponential')(z_delta_flat) # spread of latent variables 
        
        z_delta_reshape = layers.Lambda(lambda arg: K.reshape(arg, (bs, self.dim_latent, 1)))(z_delta_final) # (, dim_latent, 1)
        
        if self.latent_dist == "uniform":
            z = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                     minval=self.latent_dist_params[0], 
                                                     maxval=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
        elif self.latent_dist == "normal":
            z = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                    mean=self.latent_dist_params[0], 
                                                    stddev=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
       
        z_adjust_spread = layers.Multiply()([z_delta_reshape, z]) # (, dim_latent, n_samples)
        
        # weights
        #####
        W = layers.Flatten()(input_all)
        for l in range(self.layer_number):
            W = layers.Dense(self.nodes_number, use_bias=True, activation = 'elu')(W)
        
        W = layers.Dense(self.dim_out*self.dim_latent, use_bias=True, activation = 'linear')(W) # (, dim_out*dim_latent)
        W = layers.Reshape((self.dim_out, self.dim_latent))(W) # (, dim_out, dim_latent)
        #####
        
        z_samples = layers.Dot(axes=(2,1))([W, z_adjust_spread]) # (, dim_out, n_samples)
        y = layers.Add()([x_mean_all, z_samples])


        return Model(inputs=[input_mean, input_sd, input_all], outputs=y)
        
        
            
    def fit(self, x, y, batch_size=32, epochs=10, verbose=0, callbacks=None, validation_split=0.0, validation_data=None, sample_weight=None, learningrate=0.01):
        """
        Fits the model to traning data.

        Parameters
        ----------
        x : list of two arrays.
            x contains two arrays as inputs for the model.
            First array contains the inputs for mean model with shape (n_examples, dim_out, dim_in_mean).
            Second array contains the inputs for noise model with shape (n_examples, dim_out, dim_in_noise).
        y : array of shape (n_examples, dim_out, 1)
            Target values.
        batch_size : int, optional
            Number of samples per gradient update. The default is 32.
        epochs : int, optional
            Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
            The default is 10.
        verbose : int, optional
            0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            If not 0 will plot model summary and arachitecture as well as show the learning curve.
            The default is 0.
        callbacks : list of keras.callbacks.Callback instances, optional
            List of callbacks to apply during training. The default is None.
        validation_split : float between 0 and 1, optional
            Fraction of the training data to be used as validation data.
            The model will set apart this fraction of the training data, 
            will not train on it, and will evaluate the loss and any model 
            metrics on this data at the end of each epoch. The default is 0.0.
        validation_data : tuple of arrays like (x,y), optional
            Data on which to evaluate the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data. The default is None.
        sample_weight : array, optional
            Weights for training examples. The default is None.
        optimizer : string or keras optimizer instance, optional
            Sets options for model optimization. The default is "Adam".

        Returns
        -------
        None.

        """    

        opt = keras.optimizers.Adam(learning_rate=learningrate) # lr default 0.01
        self.model.compile(loss=EnergyScore(), optimizer=opt)
                           # experimental_run_tf_function = False)
        self.history = self.model.fit(x=x, 
                                      y=y,
                                      batch_size=batch_size, 
                                      epochs=epochs, 
                                      verbose=verbose, 
                                      callbacks=callbacks, 
                                      validation_split=validation_split, 
                                      validation_data=validation_data,
                                      shuffle=True,
                                      sample_weight=sample_weight)
            
        # create new model with same architecture and weights but 100 samples per call
        self._n_samples = 100
        weights = self.model.get_weights()
        self.model = self._build_model()
        # self.model.compile(loss=EnergyScore(), optimizer=optimizer) # not necessary if only used for prediction
        self.model.set_weights(weights)
        return self
    

    def predict(self, x_test, n_samples=1):
        """
        Draw predictive samples from model.

        Parameters
        ----------
        x_test : list of two arrays.
            x_test contains two arrays as inputs for the model.
            First array contains the inputs for mean model with shape (n_examples, dim_out, dim_in_mean).
            Second array contains the inputs for noise model with shape (n_examples, dim_out, dim_in_noise).
        n_samples : int, optional
            Number of samples to draw. The default is 1.

        Returns
        -------
        array of shape (n_examples, dim_out, n_samples)
            Predictive samples.

        """
        S = []
        for i in range(np.int(np.ceil(n_samples/self._n_samples))):
            S.append(self.model.predict(x_test))
        return np.concatenate(S, axis=2)[:, :, 0:n_samples]


    def get_model(self):
        """
        Just returns the model.

        Returns
        -------
        object
            IGEP model.

        """
        return self.model
