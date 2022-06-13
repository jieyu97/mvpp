"""
@author1: Tim Janke, Energy Information Networks & Systems Lab @ TU Darmstadt, Germany
@author2: Jieyu Chen, Chair of statistics and econometrics @ Karlsruhe Institute of Technology, Germany

Module for the implicit generative model class; The models are designed for the multivariate ensemble post-processing of wind speed forecasts;
The conditional generative model used in the paper is the Model 327 here.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.losses import Loss
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
    beta=1
    n_samples = S.shape[-1]
    def expected_dist(diff, beta):
        return K.sum(K.pow(K.sqrt(K.sum(K.square(diff), axis=-2)+K.epsilon()), beta),axis=-1)
    es_1 = expected_dist(y_true - S, beta)
    es_2 = 0
    for i in range(n_samples):
        es_2 = es_2 + expected_dist(K.expand_dims(S[:,:,i]) - S, beta)

    n_samples = tf.cast(n_samples, tf.float32)
    es = es_1/n_samples - es_2/(2*n_samples**2)
    es = tf.cast(es, tf.float32)
    return es


# subclass tensorflow.keras.losses.Loss
class EnergyScore(Loss):
    def call(self, y_true, S):
        return energy_score(y_true, S)


# the implicit generative model class for wind speed post-processing
class igep(object):
    """
    Class for conditional generative models for multivariate post-processing of wind speed forecasts.
    
    Passing model_type = 327 will create the conditional generative model used in the paper ****

    Parameters
    ----------
    dim_out : int
        Number of output dimensions.
    dim_in_mean : int
        Number of features used in the first part of inputs, for predictive mean.
    dim_in_std : int
        Number of features used in the second part of inputs, for predictive uncertainty, will be ignored in some model types.
    dim_in_features : int
        Number of features used in the third part of inputs, will be ignored in some model types.
    dim_latent : int
        Number of latent variables.
    n_samples_train : int
        Number of predictive samples to be drawn in training, for computing losses.
        More samples should results in improved accuracy but takes longer to train.
    layer_number : int
        Number of latent variables.
    nodes_number : int
        Number of latent variables.
    model_type : int, choosing from [322, 323, 325, 326, 327]
        Type of generative model selected
    latent_dist : string, optional
        Family of the latent distributions. Options are "uniform" and "normal". The default is "uniform".
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
        
        if model_type == 322:
            self._build_model = self._build_model_322
        elif model_type == 323:
            self._build_model = self._build_model_323
        elif model_type == 325:
            self._build_model = self._build_model_325
        elif model_type == 326:
            self._build_model = self._build_model_326
        elif model_type == 327:
            self._build_model = self._build_model_327
            
        self.layer_number = layer_number
        self.nodes_number = nodes_number
        
        self.model = self._build_model()
    
    

    def _build_model_322(self):        
        """
        Extended IGEP model with conditional noise;

        Returns
        -------
        object
            Keras model.

        """
        
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
                                       activation='linear')(input_mean) # [, dim_out, 1]
        
        x_mean_all = layers.Lambda(lambda arg: K.repeat_elements(arg, self._n_samples, axis=-1))(x_mean) # [, dim_out, n_samples]
        
        
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
        
        W = layers.Dense(self.dim_out*self.dim_latent, use_bias=True, activation = 'linear')(W) # [, dim_out*dim_latent]
        W = layers.Reshape((self.dim_out, self.dim_latent))(W) # [, dim_out, dim_latent]
        ##################################################################
        
        z_samples = layers.Dot(axes=(2,1))([W,z]) # [, dim_out, n_samples]
        y = layers.Add()([x_mean_all,z_samples])

        ### constraints for positive outputs
        y_positive = keras.activations.softplus(y)
        
        return Model(inputs=[input_mean, input_all], outputs=y_positive)
        
        
            
    def _build_model_323(self):        
        """
        Extended IGEP model with conditional noise;

        Returns
        -------
        object
            Keras model.

        """
        
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
                                       activation='linear')(input_mean) # [, dim_out, 1]
        
        x_mean_all = layers.Lambda(lambda arg: K.repeat_elements(arg, self._n_samples, axis=-1))(x_mean) # [, dim_out, n_samples]
        
        z_delta = layers.LocallyConnected1D(filters=1, 
                                       kernel_size=1, 
                                       strides=1,
                                       padding='valid',
                                       data_format='channels_last',
                                       use_bias=True,
                                       activation='linear')(input_sd) # [, dim_out, 1]
        
        z_delta_flat = layers.Flatten()(z_delta)
        
        z_delta_final = layers.Dense(self.dim_latent, activation = 'exponential')(z_delta_flat) # spread of latent variables 
        
        z_delta_reshape = layers.Lambda(lambda arg: K.reshape(arg, (bs, self.dim_latent, 1)))(z_delta_final) # [, dim_latent, 1]
        
        if self.latent_dist == "uniform":
            z = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                     minval=self.latent_dist_params[0], 
                                                     maxval=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
        elif self.latent_dist == "normal":
            z = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                    mean=self.latent_dist_params[0], 
                                                    stddev=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
       
        z_adjust_spread = layers.Multiply()([z_delta_reshape, z]) # [, dim_latent, n_samples]
        
        # weights
        W = layers.Flatten()(input_all)
        for l in range(self.layer_number):
            W = layers.Dense(self.nodes_number, use_bias=True, activation = 'elu')(W)
        
        W = layers.Dense(self.dim_out*self.dim_latent, use_bias=True, activation = 'linear')(W) # [, dim_out*dim_latent]
        W = layers.Reshape((self.dim_out, self.dim_latent))(W) # [, dim_out, dim_latent]
        ##################################################################
        
        z_samples = layers.Dot(axes=(2,1))([W,z_adjust_spread]) # [, dim_out, n_samples]
        y = layers.Add()([x_mean_all,z_samples])

        ### constraints for positive outputs
        y_positive = keras.activations.softplus(y)
        
        return Model(inputs=[input_mean, input_sd, input_all], outputs=y_positive)



    def _build_model_325(self):        
        """
        The conditional generative model;

        Returns
        -------
        object
            Keras model.

        """
        
        ### Inputs ###
        input_mean = keras.Input(shape=(self.dim_out, self.dim_in_mean), name = "input_mean")
        input_sd = keras.Input(shape=(self.dim_out, self.dim_in_std), name = "input_sd")
        input_all = keras.Input(shape=(self.dim_out, self.dim_in_features), name = "input_all")
        bs = K.shape(input_mean)[0]
        
        #####
        x_mean = layers.Flatten()(input_mean)
        
        x_mean = layers.Dense(25, use_bias=True, activation = 'elu')(x_mean)
        x_mean = layers.Dense(25, use_bias=True, activation = 'elu')(x_mean)
        
        x_mean = layers.Dense(self.dim_out, use_bias=True, activation = 'linear')(x_mean) # [, dim_out*1]
        x_mean = layers.Reshape((self.dim_out, 1))(x_mean) # [, dim_out, 1]
        #####
        
        x_mean_all = layers.Lambda(lambda arg: K.repeat_elements(arg, self._n_samples, axis=-1))(x_mean) # [, dim_out, n_samples]
        
        #####
        z_delta = layers.Flatten()(input_sd)
        
        z_delta = layers.Dense(25, use_bias=True, activation = 'elu')(z_delta)
        z_delta = layers.Dense(25, use_bias=True, activation = 'elu')(z_delta)
        
        z_delta = layers.Dense(self.dim_out, use_bias=True, activation = 'linear')(z_delta) # [, dim_out*1]
        #####
        
        z_delta_flat = layers.Flatten()(z_delta)
        
        z_delta_final = layers.Dense(self.dim_latent, activation = 'exponential')(z_delta_flat) # spread of latent variables 
        
        z_delta_reshape = layers.Lambda(lambda arg: K.reshape(arg, (bs, self.dim_latent, 1)))(z_delta_final) # [, dim_latent, 1]
        
        if self.latent_dist == "uniform":
            z = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                     minval=self.latent_dist_params[0], 
                                                     maxval=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
        elif self.latent_dist == "normal":
            z = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                    mean=self.latent_dist_params[0], 
                                                    stddev=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
       
        z_adjust_spread = layers.Multiply()([z_delta_reshape, z]) # [, dim_latent, n_samples]
        
        # weights
        #####
        W = layers.Flatten()(input_all)
        for l in range(self.layer_number):
            W = layers.Dense(self.nodes_number, use_bias=True, activation = 'elu')(W)
        
        W = layers.Dense(self.dim_out*self.dim_latent, use_bias=True, activation = 'linear')(W) # [, dim_out*dim_latent]
        W = layers.Reshape((self.dim_out, self.dim_latent))(W) # [, dim_out, dim_latent]
        #####
        
        z_samples = layers.Dot(axes=(2,1))([W, z_adjust_spread]) # [, dim_out, n_samples]
        y = layers.Add()([x_mean_all, z_samples])

        ### constraints for positive outputs
        y_positive = keras.activations.softplus(y)
        
        return Model(inputs=[input_mean, input_sd, input_all], outputs=y_positive)



    def _build_model_326(self):        
        """
        The conditional generative model;

        Returns
        -------
        object
            Keras model.

        """
        
        ### Inputs ###
        input_mean = keras.Input(shape=(self.dim_out, self.dim_in_mean), name = "input_mean")
        input_sd = keras.Input(shape=(self.dim_out, self.dim_in_std), name = "input_sd")
        input_all = keras.Input(shape=(self.dim_out, self.dim_in_features), name = "input_all")
        bs = K.shape(input_mean)[0]
        
        #####
        x_mean = layers.Flatten()(input_mean)
        
        x_mean = layers.Dense(25, use_bias=True, activation = 'elu')(x_mean)
        x_mean = layers.Dense(25, use_bias=True, activation = 'elu')(x_mean)
        
        x_mean = layers.Dense(self.dim_out, use_bias=True, activation = 'linear')(x_mean) # [, dim_out*1]
        x_mean = layers.Reshape((self.dim_out, 1))(x_mean) # [, dim_out, 1]
        #####
        
        x_mean_all = layers.Lambda(lambda arg: K.repeat_elements(arg, self._n_samples, axis=-1))(x_mean) # [, dim_out, n_samples]
        
        #####
        z_delta = layers.LocallyConnected1D(filters=8, 
                                       kernel_size=1, 
                                       strides=1,
                                       padding='valid',
                                       data_format='channels_last',
                                       use_bias=True,
                                       activation='linear')(input_sd) # [, dim_out, 8]
        z_delta = layers.LocallyConnected1D(filters=1, 
                                       kernel_size=1, 
                                       strides=1,
                                       padding='valid',
                                       data_format='channels_last',
                                       use_bias=True,
                                       activation='linear')(z_delta) # [, dim_out, 1]
        
        z_delta_flat = layers.Flatten()(z_delta)
        
        z_delta_final = layers.Dense(self.dim_latent, activation = 'exponential')(z_delta_flat) # spread of latent variables 
        
        z_delta_reshape = layers.Lambda(lambda arg: K.reshape(arg, (bs, self.dim_latent, 1)))(z_delta_final) # [, dim_latent, 1]
        
        if self.latent_dist == "uniform":
            z = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                     minval=self.latent_dist_params[0], 
                                                     maxval=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
        elif self.latent_dist == "normal":
            z = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                    mean=self.latent_dist_params[0], 
                                                    stddev=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
       
        z_adjust_spread = layers.Multiply()([z_delta_reshape, z]) # [, dim_latent, n_samples]
        
        # weights
        #####
        W = layers.Flatten()(input_all)
        for l in range(self.layer_number):
            W = layers.Dense(self.nodes_number, use_bias=True, activation = 'elu')(W)
        
        W = layers.Dense(self.dim_out*self.dim_latent, use_bias=True, activation = 'linear')(W) # [, dim_out*dim_latent]
        W = layers.Reshape((self.dim_out, self.dim_latent))(W) # [, dim_out, dim_latent]
        #####
        
        z_samples = layers.Dot(axes=(2,1))([W, z_adjust_spread]) # [, dim_out, n_samples]
        y = layers.Add()([x_mean_all, z_samples])

        ### constraints for positive outputs
        y_positive = keras.activations.softplus(y)
        
        return Model(inputs=[input_mean, input_sd, input_all], outputs=y_positive)
        


    def _build_model_327(self):        
        """
        The conditional generative model ysed in the paper;

        Returns
        -------
        object
            Keras model.

        """
        
        ### Inputs ###
        input_mean = keras.Input(shape=(self.dim_out, self.dim_in_mean), name = "input_mean")
        input_sd = keras.Input(shape=(self.dim_out, self.dim_in_std), name = "input_sd")
        input_all = keras.Input(shape=(self.dim_out, self.dim_in_features), name = "input_all")
        bs = K.shape(input_mean)[0]
        
        #####
        x_mean = layers.Flatten()(input_mean)
        
        x_mean = layers.Dense(100, use_bias=True, activation = 'elu')(x_mean)
        x_mean = layers.Dense(100, use_bias=True, activation = 'elu')(x_mean)
        x_mean = layers.Dense(100, use_bias=True, activation = 'elu')(x_mean)
        
        x_mean = layers.Dense(self.dim_out, use_bias=True, activation = 'elu')(x_mean) # [, dim_out*1]
        x_mean = layers.Reshape((self.dim_out, 1))(x_mean) # [, dim_out, 1]
        #####
        
        x_mean_all = layers.Lambda(lambda arg: K.repeat_elements(arg, self._n_samples, axis=-1))(x_mean) # [, dim_out, n_samples]
        
        #####
        z_delta = layers.Flatten()(input_sd)
        
        z_delta = layers.Dense(100, use_bias=True, activation = 'elu')(z_delta)
        z_delta = layers.Dense(100, use_bias=True, activation = 'elu')(z_delta)
        z_delta = layers.Dense(100, use_bias=True, activation = 'elu')(z_delta)
        
        z_delta_final = layers.Dense(self.dim_latent, activation = 'exponential')(z_delta) # spread of latent variables 
        z_delta_reshape = layers.Reshape((self.dim_latent, 1))(z_delta_final) # [, dim_latent, 1]
        #####
        
        if self.latent_dist == "uniform":
            z = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                     minval=self.latent_dist_params[0], 
                                                     maxval=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
        elif self.latent_dist == "normal":
            z = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                    mean=self.latent_dist_params[0], 
                                                    stddev=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
       
        z_adjust_spread = layers.Multiply()([z_delta_reshape, z]) # [, dim_latent, n_samples]
        
        #####
        W = layers.Flatten()(input_all)
        z_n = layers.Flatten()(z_adjust_spread)
        W = layers.Concatenate(axis=1)([W, z_n])
        
        W = layers.Dense(25, use_bias=True, activation = 'elu')(W)
        W = layers.Dense(25, use_bias=True, activation = 'elu')(W)
        
        W = layers.Dense(self.dim_out*self._n_samples, use_bias=True, activation = 'linear')(W) # [, dim_out*n_samples]
        z_samples = layers.Reshape((self.dim_out, self._n_samples))(W) # [, dim_out, n_samples]
        
        y = layers.Add()([x_mean_all, z_samples])

        ### constraints for positive outputs
        y_positive = keras.activations.softplus(y)
             
        return Model(inputs=[input_mean, input_sd, input_all], outputs=y_positive)

        
        
    def fit(self, x, y, batch_size=32, epochs=10, verbose=0, callbacks=None, 
            validation_split=0.0, validation_data=None, sample_weight=None, 
            learningrate=0.01):
        """
        Fits the model to traning data.

        Parameters
        ----------
        x : list of arrays.
            Inputs for the model, in the training set.
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
            List of callbacks to apply during training. 
            The default is None.
        validation_split : float between 0 and 1, optional
            Fraction of the training data to be used as validation data.
            The model will set apart this fraction of the training data, 
            will not train on it, and will evaluate the loss and any model 
            metrics on this data at the end of each epoch. 
            The default is 0.0.
        validation_data : tuple of arrays like (x,y), optional
            Data on which to evaluate the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data. 
            The default is None.
        sample_weight : array, optional
            Weights for training examples. 
            The default is None.
        optimizer : string or keras optimizer instance, optional
            Sets options for model optimization. 
            The default is "Adam".

        Returns
        -------
        None.

        """    

        opt = keras.optimizers.Adam(learning_rate=learningrate)
        self.model.compile(loss=EnergyScore(), optimizer=opt)
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

        return self
    

    def predict(self, x_test, n_samples=1):
        """
        Draw predictive samples from model.

        Parameters
        ----------
        x_test : list of arrays.
            Inputs for the model, in the test set.
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
        Return the model.

        Returns
        -------
        object
            Generative model.

        """
        return self.model
