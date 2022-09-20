"""
@author: Tim Janke, Energy Information Networks & Systems Lab @ TU Darmstadt, Germany
@author: Jieyu Chen, ECON @ KIT, Germany

Module for CGM class.
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
def energy_score(y_true, y_pred):
    """
    Computes energy score efficiently.
    Parameters
    ----------
    y_true : tf tensor of shape (BATCH_SIZE, D, 1)
        True values.
    y_pred : tf tensor of shape (BATCH_SIZE, D, N_SAMPLES)
        Predictive samples.
    Returns
    -------
    tf tensor of shape (BATCH_SIZE,)
        Scores.
    """
    n_samples_model = tf.cast(tf.shape(y_pred)[2], dtype=tf.float32)
    
    es_12 = tf.reduce_sum(tf.sqrt(tf.clip_by_value(tf.matmul(y_true, y_true, transpose_a=True, transpose_b=False) + 
                                                   tf.square(tf.linalg.norm(y_pred, axis=1, keepdims=True)) - 
                                                   2*tf.matmul(y_true, y_pred, transpose_a=True, transpose_b=False), 
                                                   K.epsilon(), 1e10)
                                  ), 
                          axis=(1,2))    
    G = tf.linalg.matmul(y_pred, y_pred, transpose_a=True, transpose_b=False)
    d = tf.expand_dims(tf.linalg.diag_part(G, k=0), axis=1)
    es_22 = tf.reduce_sum(tf.sqrt(tf.clip_by_value(d + tf.transpose(d, perm=(0,2,1)) - 2*G, 
                                                   K.epsilon(), 1e10)
                                  ), 
                          axis=(1,2))

    loss = es_12/(n_samples_model) -  es_22/(2*n_samples_model*(n_samples_model-1))
    
    return loss


# subclass Keras loss
class EnergyScore(Loss):
    def __init__(self, name="EnergyScore", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_data, y_model):
        return energy_score(y_data, y_model)


class cgm(object):
    """
    Class for conditional generative models (CGMs).
    
    Passing model_type = 't2m' will create model for temperature forecasts.
    Passing model_type = 'ws' will create model for wind speed forecasts.

    Parameters
    ----------
    dim_out : int
        Number of output dimensions.
    dim_in_mean : int
        Number of features used for predictive mean.
    dim_in_std : int
        Number of features used for uncertainty. 
    dim_in_features : int
        Number of all features used. 
    dim_latent : int
        Number of latent variables.
    n_samples_train : int
        Number of predictive samples to be drawn in training.
        More samples should results in improved accuracy but takes longer to train.
    latent_dist : string, optional
        Family of the latent distributions. Options are uniform and normal.
    latent_dist_params : tuple, optional
        Parameters for latent distributions. (min, max) for uniform, (mean, stddev) for normal. 
        If None is passed params are set to (-1, 1) and (0, 1) respectively.

    Returns
    -------
    None.

    """
    
    def __init__(self, model_type, dim_out, dim_in_mean, dim_in_std, dim_in_features, 
                 dim_latent, n_samples_train, latent_dist, latent_dist_params=None):
        
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
        
        if model_type == 't2m':
            self._build_model = self._build_model_t2m
        elif model_type == 'ws':
            self._build_model = self._build_model_ws
    
        self.model = self._build_model()
    
        
    def _build_model_t2m(self):        

        ### Inputs ###
        input_mean = keras.Input(shape=(self.dim_out, self.dim_in_mean), name = "input_mean")
        input_std = keras.Input(shape=(self.dim_out, self.dim_in_std), name = "input_std")
        input_all = keras.Input(shape=(self.dim_out, self.dim_in_features), name = "input_all")
        bs = tf.shape(input_mean)[0]
        
        ##### Mean model ####
#        y_mean = layers.LocallyConnected1D(filters=1, 
#                                       kernel_size=1, 
#                                       strides=1,
#                                       padding='valid',
#                                       data_format='channels_last',
#                                       use_bias=True,
#                                       activation='linear')(input_mean) # (, dim_out, 1)
#        y_mean = layers.Flatten()(y_mean) # (, dim_out)
        
        y_mean = layers.Flatten()(input_mean) # (, dim_out*dim_in_mean)
        
        y_mean = layers.Dense(self.dim_out, use_bias=True, activation = 'linear')(y_mean) # (, dim_out)
        
        y_mean = layers.RepeatVector(self.n_samples_train)(y_mean) # (, n_samples, dim_out)
        y_mean = layers.Permute((2,1))(y_mean) # (, dim_out, n_samples)

        ##### Conditional noise model ####
        ### Noise part
        delta_z = layers.Flatten()(input_std) # (, dim_out*dim_in_std)
        
        delta_z = layers.Dense(self.dim_latent, activation = 'exponential')(delta_z) # (, dim_latent)
        
        delta_z = layers.Reshape((self.dim_latent, 1))(delta_z) # (, dim_latent, 1)
        # scale of latent variables

        if self.latent_dist == "uniform":
            epsilon = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                     minval=self.latent_dist_params[0], 
                                                     maxval=self.latent_dist_params[1]))([bs, self.dim_latent, self.n_samples_train])
        elif self.latent_dist == "normal":
            epsilon = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                    mean=self.latent_dist_params[0], 
                                                    stddev=self.latent_dist_params[1]))([bs, self.dim_latent, self.n_samples_train])
       
        z = layers.Multiply()([delta_z, epsilon]) # (, dim_latent, n_samples)
        
        ### Weights part
        all_predictors = layers.Flatten()(input_all) # (, dim_out*dim_in_features)
        all_predictors = layers.RepeatVector(self.n_samples_train)(all_predictors) # (, n_samples, dim_out*dim_in_features)
        all_predictors = layers.Permute((2,1))(all_predictors) # (, dim_out*dim_in_features, n_samples)
        
        W = layers.Concatenate(axis=1)([all_predictors, z]) # (, dim_out*dim_in_festures+dim_latent, n_samples)
        W = layers.Permute((2,1))(W) # (, n_samples, dim_out*dim_in_festures+dim_latent)
        # Dense is applied to last dimension, i.e. we need features in last dim
        
        W = layers.Dense(100, use_bias=True, activation = 'elu')(W)
        W = layers.Dense(100, use_bias=True, activation = 'elu')(W)
        
        W = layers.Dense(self.dim_out, use_bias=True, activation = 'linear')(W) # (, n_samples, dim_out)
        
        y_noise = layers.Permute((2,1))(W) # (, dim_out, n_samples)
        
        y = layers.Add()([y_mean, y_noise])

        return Model(inputs=[input_mean, input_std, input_all], outputs=y)



    def _build_model_ws(self):        

        ### Inputs ###
        input_mean = keras.Input(shape=(self.dim_out, self.dim_in_mean), name = "input_mean")
        input_std = keras.Input(shape=(self.dim_out, self.dim_in_std), name = "input_std")
        input_all = keras.Input(shape=(self.dim_out, self.dim_in_features), name = "input_all")
        bs = tf.shape(input_mean)[0]
        
        ##### Mean model ####
#        y_mean = layers.LocallyConnected1D(filters=1, 
#                                        kernel_size=1, 
#                                        strides=1,
#                                        padding='valid',
#                                        data_format='channels_last',
#                                        use_bias=True,
#                                        activation='linear')(input_mean) # (, dim_out, 1)
#        y_mean = layers.Flatten()(y_mean) # (, dim_out)
        
        y_mean = layers.Flatten()(input_mean) # (, dim_out*dim_in_mean)
        
        y_mean = layers.Dense(100, use_bias=True, activation = 'elu')(y_mean)
        y_mean = layers.Dense(100, use_bias=True, activation = 'elu')(y_mean)
        # y_mean = layers.Dense(100, use_bias=True, activation = 'elu')(y_mean)
        
        y_mean = layers.Dense(self.dim_out, use_bias=True, activation = 'elu')(y_mean) # (, dim_out)
        
        y_mean = layers.RepeatVector(self.n_samples_train)(y_mean) # (, n_samples, dim_out)
        y_mean = layers.Permute((2,1))(y_mean) # (, dim_out, n_samples)

        ##### Conditional noise model ####
        ### Noise part
        delta_z = layers.Flatten()(input_std) # (, dim_out*dim_in_std)
        
        delta_z = layers.Dense(100, use_bias=True, activation = 'elu')(delta_z)
        delta_z = layers.Dense(100, use_bias=True, activation = 'elu')(delta_z)
        # delta_z = layers.Dense(100, use_bias=True, activation = 'elu')(delta_z)
        
        delta_z = layers.Dense(self.dim_latent, activation = 'exponential')(delta_z) # (, dim_latent)
        
        delta_z = layers.Reshape((self.dim_latent, 1))(delta_z) # (, dim_latent, 1)
        # scale of latent variables

        if self.latent_dist == "uniform":
            epsilon = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                     minval=self.latent_dist_params[0], 
                                                     maxval=self.latent_dist_params[1]))([bs, self.dim_latent, self.n_samples_train])
        elif self.latent_dist == "normal":
            epsilon = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                    mean=self.latent_dist_params[0], 
                                                    stddev=self.latent_dist_params[1]))([bs, self.dim_latent, self.n_samples_train])
       
        z = layers.Multiply()([delta_z, epsilon]) # (, dim_latent, n_samples)
        
        ### Weights part
        all_predictors = layers.Flatten()(input_all) # (, dim_out*dim_in_features)
        all_predictors = layers.RepeatVector(self.n_samples_train)(all_predictors) # (, n_samples, dim_out*dim_in_features)
        all_predictors = layers.Permute((2,1))(all_predictors) # (, dim_out*dim_in_features, n_samples)
        
        W = layers.Concatenate(axis=1)([all_predictors, z]) # (, dim_out*dim_in_festures+dim_latent, n_samples)
        W = layers.Permute((2,1))(W) # (, n_samples, dim_out*dim_in_festures+dim_latent)
        # Dense is applied to last dimension, i.e. we need features in last dim
        
        W = layers.Dense(100, use_bias=True, activation = 'elu')(W)
        W = layers.Dense(100, use_bias=True, activation = 'elu')(W)
        
        W = layers.Dense(self.dim_out, use_bias=True, activation = 'linear')(W) # (, n_samples, dim_out)
        
        y_noise = layers.Permute((2,1))(W) # (, dim_out, n_samples)
        
        y = layers.Add()([y_mean, y_noise])

        #### force positive outputs
        y_positive = keras.activations.softplus(y)
        
        return Model(inputs=[input_mean, input_std, input_all], outputs=y_positive)
    

            
    def fit(self, x, y, batch_size=64, epochs=300, verbose=0, callbacks=None, 
            validation_split=0.0, validation_data=None, sample_weight=None, learningrate=0.01):
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
        for i in range(np.int(np.ceil(n_samples/self.n_samples_train))):
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