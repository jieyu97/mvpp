
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import pandas as pd
from scoringRules import es_sample 
from sklearn import preprocessing
plt.close("all")

# from hyperopt.pyll.base import scope 
from hyperopt import Trials, tpe, fmin, hp, STATUS_OK

# import pickle

import tensorflow.compat.v1 as tfv
tfv.disable_v2_behavior()

# fix random seed for reproducibility
seed = 8
np.random.seed(seed)

# set model hyper parameters
N_SAMPLES_TRAIN = 50     # number of samples drawn during training
EPOCHS = 100       # max epochs 
# misc options
VERBOSE = 2                     # determines how much info is given about fitting
PLOT_LEARNING_CURVE = False     # if True, the model's learning curve is plotted
N_SAMPLES_TEST = 50 # 1000           # number of samples used for computung scores
DIM = 10                        # dimension of target values
TAUS = np.linspace(1,99,99)/100 # which quantiles to evaluate
PLOT_RESULTS = False

# Model 3.2.3 hyperparameters
latent_dist = 'normal'
learningrate = 0.001
epochs = EPOCHS
# important hp
batch_size = 64
layer_number = 2
nodes_number = 25
dim_latent = 26



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


path = '/home/chen_jieyu/IGEP/ECMWF_wind_data.feather'
t2m_ens_complete = pd.read_feather(path)

path_add = '/home/chen_jieyu/IGEP/wind_additional_predictors.feather'
t2m_add_complete = pd.read_feather(path_add)

dist_samples = pd.read_csv('/home/chen_jieyu/IGEP/ws_dist_10samples.csv', header=None)

for k in [10,20,30,40,50,60,70,80,90]:

    tfv.reset_default_graph()
    
    station_sample = dist_samples.iloc[k,]
    ens_sample = t2m_ens_complete[t2m_ens_complete['station'].isin(station_sample)]
    dateobs_count = ens_sample.groupby('date')['date'].count()
    dates = dateobs_count.index
    used_dates = dates[dateobs_count == DIM]
    used_ens_sample = ens_sample[ens_sample['date'].isin(used_dates)]
    
    add_sample = t2m_add_complete[t2m_add_complete['station'].isin(station_sample)]
    used_add_sample = add_sample[add_sample['date'].isin(used_dates)]
    
    # t2m data
    t2m_obs = used_ens_sample['obs']
    t2m_obs.index = used_ens_sample['date']
    data_obs = t2m_obs
    
    # set initial training and test dates
    train_dateindex = ((t2m_obs.index.year != 2016) & (t2m_obs.index.year != 2015))
    val_dateindex = (t2m_obs.index.year == 2015)
    test_dateindex = (t2m_obs.index.year == 2016)
    
    # Predictions
    t2m_ens = used_ens_sample.iloc[:, 3:53]
    t2m_ens.index = used_ens_sample['date']
    data_ens = t2m_ens
    
    # added predictors
    add_dim = 42
    t2m_add = used_add_sample.loc[:,["sp_mean","sp_var",
                                     "u_pl850_mean","u_pl850_var",
                                     "v_pl850_mean","v_pl850_var",
                                     "ws_pl850_mean","ws_pl850_var",
                                     "q_pl850_mean","q_pl850_var",
                                     "u_pl500_mean","u_pl500_var",
                                     "v_pl500_mean","v_pl500_var",
                                     "ws_pl500_mean","ws_pl500_var",
                                     "u10_mean","u10_var",
                                     "v10_mean","v10_var",
                                     "t2m_mean","t2m_var",
                                     "d2m_mean","d2m_var",
                                     "cape_mean","cape_var",
                                     "tcc_mean","tcc_var",
                                     "gh_pl500_mean","gh_pl500_var",
                                     "sshf_mean","sshf_var",
                                     "slhf_mean","slhf_var",
                                     "ssr_mean","ssr_var",
                                     "str_mean","str_var",
                                 "lat","lon","alt","sin_yday"]]
    t2m_add.index = used_add_sample['date']
    data_add = t2m_add
    
    # get training and test data
    obser = data_obs.copy()
    pred = data_ens.copy()
    addpre = data_add.copy()
    dim = DIM
    
    
    ######### standardization
    #    scaler = preprocessing.StandardScaler().fit(obser[train_dateindex].values.reshape(-1,1))
    #    stand_obs = scaler.transform(obser.values.reshape(-1,1)).reshape(-1)
    #    obser.iloc[:] = stand_obs
    #    
    #    for i in range(pred.shape[1]):
    #        pred.iloc[:,i] = scaler.transform(pred.iloc[:,i].values.reshape(-1,1))
    
    norm_obs = obser.copy()
    norm_pred = pred.copy()
    
    scaler = preprocessing.MinMaxScaler().fit(norm_obs[train_dateindex].values.reshape(-1,1)) 
    stand_obs = scaler.transform(norm_obs.values.reshape(-1,1)).reshape(-1)
    norm_obs.iloc[:] = stand_obs
    
    for i in range(norm_pred.shape[1]):
        norm_pred.iloc[:,i] = scaler.transform(norm_pred.iloc[:,i].values.reshape(-1,1))
    
    norm_ens_mu = norm_pred.mean(axis=1)
    norm_ens_sigma = norm_pred.std(axis=1)
    
    norm_ens_max = norm_pred.max(axis=1)
    norm_ens_min = norm_pred.min(axis=1)
    norm_ens_spread = norm_ens_max - norm_ens_min
    
    ens_mu = pred.mean(axis=1)
    ens_sigma = pred.var(axis=1)
    
    ens_max = pred.max(axis=1)
    ens_min = pred.min(axis=1)
    ens_spread = ens_max - ens_min
    
    for i in range(addpre.shape[1]-1):
        scaler_i = preprocessing.StandardScaler().fit(addpre.iloc[train_dateindex,i].values.reshape(-1,1))
        addpre.iloc[:,i] = scaler_i.transform(addpre.iloc[:,i].values.reshape(-1,1))
        
    
    add_pre_mu = addpre.loc[:,["sp_mean", "u_pl850_mean", "v_pl850_mean", "ws_pl850_mean", "q_pl850_mean",
                               "u_pl500_mean", "v_pl500_mean", "ws_pl500_mean", "u10_mean", "v10_mean",
                               "t2m_mean", "d2m_mean", "cape_mean", "tcc_mean", "gh_pl500_mean",
                               "sshf_mean", "slhf_mean", "ssr_mean", "str_mean", "sin_yday"]]
    
    add_pre_sigma = addpre.loc[:,["sp_var","u_pl850_var","v_pl850_var","ws_pl850_var","q_pl850_var",
                                  "u_pl500_var", "v_pl500_var", "ws_pl500_var", "u10_var", "v10_var",
                                  "t2m_var", "d2m_var", "cape_var", "tcc_var", "gh_pl500_var",
                                  "sshf_var", "slhf_var", "ssr_var", "str_var", "sin_yday"]]
    
    n_add = 20
    
    # Inputs
    x_train_m323 = [np.concatenate((ens_mu[train_dateindex].values.reshape((-1, dim, 1)),
                                    add_pre_mu[train_dateindex].values.reshape((-1, dim, n_add))
                                    ), axis=-1),
                    np.concatenate((ens_sigma[train_dateindex].values.reshape((-1, dim, 1)),
                                    add_pre_sigma[train_dateindex].values.reshape((-1, dim, n_add))
                                    ), axis=-1),
                    np.concatenate((norm_ens_mu[train_dateindex].values.reshape((-1, dim, 1)),
                                    norm_ens_sigma[train_dateindex].values.reshape((-1, dim, 1)),
                                    norm_ens_max[train_dateindex].values.reshape((-1, dim, 1)),
                                    norm_ens_min[train_dateindex].values.reshape((-1, dim, 1)),
                                    norm_ens_spread[train_dateindex].values.reshape((-1, dim, 1)),
                                    addpre[train_dateindex].values.reshape((-1, dim, add_dim))
                                    ), axis=-1)]    
    x_val_m323 = [np.concatenate((ens_mu[val_dateindex].values.reshape((-1, dim, 1)),
                                   add_pre_mu[val_dateindex].values.reshape((-1, dim, n_add))
                                    ), axis=-1),
                    np.concatenate((ens_sigma[val_dateindex].values.reshape((-1, dim, 1)),
                                    add_pre_sigma[val_dateindex].values.reshape((-1, dim, n_add))
                                    ), axis=-1),
                    np.concatenate((norm_ens_mu[val_dateindex].values.reshape((-1, dim, 1)),
                                    norm_ens_sigma[val_dateindex].values.reshape((-1, dim, 1)),
                                    norm_ens_max[val_dateindex].values.reshape((-1, dim, 1)),
                                    norm_ens_min[val_dateindex].values.reshape((-1, dim, 1)),
                                    norm_ens_spread[val_dateindex].values.reshape((-1, dim, 1)),
                                    addpre[val_dateindex].values.reshape((-1, dim, add_dim))
                                    ), axis=-1)]   
    x_test_m323 = [np.concatenate((ens_mu[test_dateindex].values.reshape((-1, dim, 1)),
                                   add_pre_mu[test_dateindex].values.reshape((-1, dim, n_add))
                                    ), axis=-1),
                    np.concatenate((ens_sigma[test_dateindex].values.reshape((-1, dim, 1)),
                                    add_pre_sigma[test_dateindex].values.reshape((-1, dim, n_add))
                                    ), axis=-1),
                    np.concatenate((norm_ens_mu[test_dateindex].values.reshape((-1, dim, 1)),
                                    norm_ens_sigma[test_dateindex].values.reshape((-1, dim, 1)),
                                    norm_ens_max[test_dateindex].values.reshape((-1, dim, 1)),
                                    norm_ens_min[test_dateindex].values.reshape((-1, dim, 1)),
                                    norm_ens_spread[test_dateindex].values.reshape((-1, dim, 1)),
                                    addpre[test_dateindex].values.reshape((-1, dim, add_dim))
                                    ), axis=-1)]        
         
    
    y_train = obser[train_dateindex].values.reshape((-1, dim, 1))
    y_val = obser[val_dateindex].values.reshape((-1, dim, 1))
    y_test = obser[test_dateindex].values.reshape((-1, dim, 1))
    
    y_train_tmp = y_train
    y_val_tmp = y_val
    y_test_tmp = y_test
    
    testy = data_obs[test_dateindex]
    
    
    
    # model wrapping
    # create model, fit and train model, test model, and get the ES output.
            
    dim_out = DIM
    dim_in_features_mu = x_train_m323[0].shape[-1]
    dim_in_features_sigma = x_train_m323[1].shape[-1]
    dim_in_features_all = x_train_m323[2].shape[-1] # dim_in_features_all != dim_in_features*2 here
    n_samples = N_SAMPLES_TRAIN
    
    train_x = x_train_m323
    train_y = y_train_tmp
    val_x = x_val_m323
    val_y = y_val_tmp
    test_x = x_test_m323
    test_y = testy
    
    
    if latent_dist == "uniform":
        latent_dist_params = (-1.0, 1.0)
    elif latent_dist == "normal":
        latent_dist_params = (0.0, 1.0)
    
    
    ### Inputs ###
    input_mean = keras.Input(shape=(dim_out, dim_in_features_mu), name = "input_mean")
    input_sd = keras.Input(shape=(dim_out, dim_in_features_sigma), name = "input_sd")
    input_all = keras.Input(shape=(dim_out, dim_in_features_all), name = "input_all")
    bs = K.shape(input_mean)[0]
    
    ##########
    x_mean = layers.Flatten()(input_mean) # (, dim_out*dim_in_features)
    
    x_mean = layers.Dense(100, use_bias=True, activation = 'elu')(x_mean)
    x_mean = layers.Dense(100, use_bias=True, activation = 'elu')(x_mean)
    x_mean = layers.Dense(100, use_bias=True, activation = 'elu')(x_mean)
    
    x_mean_flat = layers.Dense(dim_out, use_bias=True, activation = 'elu')(x_mean) # (, dim_out)
    ##########
    
    #x_mean = layers.LocallyConnected1D(filters=1, 
    #                               kernel_size=1, 
    #                               strides=1,
    #                               padding='valid',
    #                               data_format='channels_last',
    #                               use_bias=True,
    #                               activation='linear')(input_mean) # (, dim_out, 1)
    #
    #x_mean_flat = layers.Flatten()(x_mean) # (, dim_out)
    #
    #x_mean_flat = layers.Dense(dim_out, activation = 'linear')(x_mean_flat) # (, dim_out)
    
    x_mean_final = layers.Reshape((dim_out, 1))(x_mean_flat) # (, dim_out, 1)
    
    x_mean_all = layers.Lambda(lambda arg: K.repeat_elements(arg, n_samples, axis=-1))(x_mean_final) # (, dim_out, n_samples)
    
    ###########
    z_delta = layers.Flatten()(input_sd) # (, dim_out*dim_in_features)
    
    z_delta = layers.Dense(100, use_bias=True, activation = 'elu')(z_delta)
    z_delta = layers.Dense(100, use_bias=True, activation = 'elu')(z_delta)
    z_delta = layers.Dense(100, use_bias=True, activation = 'elu')(z_delta)
    
#    z_delta = layers.Dense(dim_out, use_bias=True, activation = 'linear')(z_delta) # (, dim_out)
    
    z_delta_final = layers.Dense(dim_latent, use_bias=True, activation = 'exponential')(z_delta) # (, dim_latent)
    ###########
    
    #z_delta = layers.LocallyConnected1D(filters=1, 
    #                               kernel_size=1, 
    #                               strides=1,
    #                               padding='valid',
    #                               data_format='channels_last',
    #                               use_bias=True,
    #                               activation='linear')(input_sd) # (, dim_out, 1)
    #
    #z_delta_flat = layers.Flatten()(z_delta)
    #
    #z_delta_final = layers.Dense(dim_latent, activation = 'exponential')(z_delta_flat) # spread of latent variables 
    
    z_delta_reshape = layers.Reshape((dim_latent, 1))(z_delta_final) # (, dim_latent, 1)
    
    
    if latent_dist == "uniform":
        z = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                 minval=latent_dist_params[0], 
                                                 maxval=latent_dist_params[1]))([bs, dim_latent, n_samples])
    elif latent_dist == "normal":
        z = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                mean=latent_dist_params[0], 
                                                stddev=latent_dist_params[1]))([bs, dim_latent, n_samples])
    
    z_adjust_spread = layers.Multiply()([z_delta_reshape, z]) # (, dim_latent, n_samples)
    
    # weights
#    W = layers.Flatten()(input_all)
#    for l in range(layer_number):
#        W = layers.Dense(nodes_number, use_bias=True, activation = 'elu')(W)
#    
#    W = layers.Dense(dim_out*dim_latent, use_bias=True, activation = 'linear')(W) # (, dim_out*dim_latent)
#    W = layers.Reshape((dim_out, dim_latent))(W) # (, dim_out, dim_latent)
#    ##################################################################
#    
#    z_samples = layers.Dot(axes=(2,1))([W,z_adjust_spread]) # (, dim_out, n_samples)
#    y = layers.Add()([x_mean_all,z_samples])    
    
    
    W = layers.Flatten()(input_all)
    z_n = layers.Flatten()(z_adjust_spread)
    W_con = layers.Concatenate(axis=1)([W, z_n])
    
#    W_con = layers.Dense(100, use_bias=True, activation = 'elu')(W_con)
#    W_con = layers.Dense(100, use_bias=True, activation = 'elu')(W_con)
    W_con = layers.Dense(100, use_bias=True, activation = 'elu')(W_con)
        
    W_con = layers.Dense(dim_out*n_samples, use_bias=True, activation = 'linear')(W_con) # (, dim_out*n_samples)
    z_samples = layers.Reshape((dim_out, n_samples))(W_con) # (, dim_out, n_samples)
    y = layers.Add()([x_mean_all, z_samples]) 
    
    #### force positive outputs
    y_positive = keras.activations.softplus(y)
    
    model = Model(inputs = [input_mean, input_sd, input_all], outputs = y_positive)
    
    
    opt = keras.optimizers.Adam(learning_rate = learningrate) # lr default 0.01
    model.compile(loss = EnergyScore(), optimizer = opt)
    
    callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.001, patience = 3,
                                                restore_best_weights = True)
    model.fit(x = train_x, y = train_y,
              batch_size = batch_size, 
              epochs = epochs, 
              verbose = 0, 
              callbacks = [callback], 
              validation_split = 0.0, 
              validation_data = (val_x, val_y),        
              sample_weight=None)
    
    N_SAMPLES_TEST = 50
    
    # predict and append to list
    S_m3 = []
    # S_m3.append(scaler.inverse_transform(model.predict(x_test_m323, N_SAMPLES_TEST)))
    S_m3.append(model.predict(x_test_m323, N_SAMPLES_TEST))
    
    S_m3 = np.concatenate(S_m3, axis = 0)
    
    ES = es_sample(y = np.reshape(test_y.values, (-1, DIM)), dat = S_m3)
    
    pd_output = pd.DataFrame(np.reshape(S_m3, (S_m3.shape[0]*S_m3.shape[1],-1)), index=testy.index)
        
    print(pd_output)
        
    print('IGEP - wind speed - MVPP - energy score:')
    print(k)
    print(ES)







