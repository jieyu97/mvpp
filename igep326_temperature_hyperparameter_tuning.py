
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
import tensorflow.compat.v1 as tfv
tfv.disable_v2_behavior()

# from hyperopt.pyll.base import scope 
from hyperopt import Trials, tpe, fmin, hp, STATUS_OK

import pickle

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
DIM = 5                        # dimension of target values
TAUS = np.linspace(1,99,99)/100 # which quantiles to evaluate
PLOT_RESULTS = False

# Read data
path = '/home/chen_jieyu/IGEP/ens_fc_t2m_complete.feather'
t2m_ens_complete = pd.read_feather(path)

path_add = '/home/chen_jieyu/IGEP/tem_additional_predictors.feather'
t2m_add_complete = pd.read_feather(path_add)

dist_samples = pd.read_csv('/home/chen_jieyu/IGEP/dist_5samples.csv', header = None)

k=0

station_sample = dist_samples.iloc[k,]

ens_sample = t2m_ens_complete[t2m_ens_complete['station'].isin(station_sample)]
# ens_sample.info()
dateobs_count = ens_sample.groupby('date')['date'].count()
dates = dateobs_count.index
used_dates = dates[dateobs_count == DIM]
used_ens_sample = ens_sample[ens_sample['date'].isin(used_dates)]

add_sample = t2m_add_complete[t2m_add_complete['station'].isin(station_sample)]
used_add_sample = add_sample[add_sample['date'].isin(used_dates)]

# LOAD DATA
# t2m data
t2m_obs = used_ens_sample['obs']
t2m_obs.index = used_ens_sample['date']
data_obs = t2m_obs

# set initial training and test dates
train_dateindex = ((t2m_obs.index.year != 2016) & (t2m_obs.index.year != 2015))
val_dateindex = (t2m_obs.index.year == 2015)
test_dateindex = (t2m_obs.index.year == 2016)

# Predictions
t2m_ens = used_ens_sample.iloc[:,3:53]
t2m_ens.index = used_ens_sample['date']
data_ens = t2m_ens

# added predictors
add_dim = 37
t2m_add = used_add_sample.loc[:,["d2m_mean","d2m_var",
                                 "q_pl850_mean","q_pl850_var",
                                 "tcc_mean","tcc_var",
                                 "u_pl850_mean","u_pl850_var",
                                 "v_pl850_mean","v_pl850_var",
                                 "sshf_mean","sshf_var",
                                 "slhf_mean","slhf_var",
                                 "u10_mean","u10_var",
                                 "v10_mean","v10_var",
                                 "cape_mean","cape_var",
                                 "sp_mean","sp_var",
                                 "u_pl500_mean","u_pl500_var",
                                 "v_pl500_mean","v_pl500_var",
                                 "gh_pl500_mean","gh_pl500_var",
                                 "ssr_mean","ssr_var",
                                 "str_mean","str_var",
                                 "lat","lon","alt","orog","sin_yday"]]
t2m_add.index = used_add_sample['date']
data_add = t2m_add


# SPLIT DATA

# get training and test data
obser = data_obs.copy()
pred = data_ens.copy()
addpre = data_add.copy()
dim = DIM


######### standardization
scaler = preprocessing.StandardScaler().fit(obser[train_dateindex].values.reshape(-1,1))
stand_obs = scaler.transform(obser.values.reshape(-1,1)).reshape(-1)
obser.iloc[:] = stand_obs

for i in range(pred.shape[1]):
    pred.iloc[:,i] = scaler.transform(pred.iloc[:,i].values.reshape(-1,1))

ens_mu = pred.mean(axis=1)
ens_sigma = pred.var(axis=1)

ens_max = pred.max(axis=1)
ens_min = pred.min(axis=1)
ens_spread = ens_max - ens_min

for i in range(addpre.shape[1]-1):
    scaler_i = preprocessing.StandardScaler().fit(addpre.iloc[train_dateindex,i].values.reshape(-1,1))
    addpre.iloc[:,i] = scaler_i.transform(addpre.iloc[:,i].values.reshape(-1,1))

add_pre_mu = addpre.loc[:,["d2m_mean","q_pl850_mean","tcc_mean","u_pl850_mean","v_pl850_mean",
                           "sshf_mean","slhf_mean","u10_mean","v10_mean","cape_mean","sp_mean",
                           "u_pl500_mean","v_pl500_mean","gh_pl500_mean","ssr_mean","str_mean"]]

add_pre_sigma = addpre.loc[:,["d2m_var","q_pl850_var","tcc_var","u_pl850_var","v_pl850_var",
                              "sshf_var","slhf_var","u10_var","v10_var","cape_var","sp_var",
                              "u_pl500_var","v_pl500_var","gh_pl500_var","ssr_var","str_var"]]

n_add = 16

# Inputs for Model 3.2.3
x_train_m323 = [np.concatenate((ens_mu[train_dateindex].values.reshape((-1, dim, 1)),
                                add_pre_mu[train_dateindex].values.reshape((-1, dim, n_add))
                                ), axis=-1),
                np.concatenate((ens_sigma[train_dateindex].values.reshape((-1, dim, 1)),
                                ens_max[train_dateindex].values.reshape((-1, dim, 1)),
                                ens_min[train_dateindex].values.reshape((-1, dim, 1)),
                                ens_spread[train_dateindex].values.reshape((-1, dim, 1)),
                                add_pre_sigma[train_dateindex].values.reshape((-1, dim, n_add))
                                ), axis=-1),
                np.concatenate((ens_mu[train_dateindex].values.reshape((-1, dim, 1)),
                                ens_sigma[train_dateindex].values.reshape((-1, dim, 1)),
                                ens_max[train_dateindex].values.reshape((-1, dim, 1)),
                                ens_min[train_dateindex].values.reshape((-1, dim, 1)),
                                ens_spread[train_dateindex].values.reshape((-1, dim, 1)),
                                addpre[train_dateindex].values.reshape((-1, dim, add_dim))
                                ), axis=-1)]    
x_val_m323 = [np.concatenate((ens_mu[val_dateindex].values.reshape((-1, dim, 1)),
                              add_pre_mu[val_dateindex].values.reshape((-1, dim, n_add))
                                ), axis=-1),
                np.concatenate((ens_sigma[val_dateindex].values.reshape((-1, dim, 1)),
                                ens_max[val_dateindex].values.reshape((-1, dim, 1)),
                                ens_min[val_dateindex].values.reshape((-1, dim, 1)),
                                ens_spread[val_dateindex].values.reshape((-1, dim, 1)),
                                add_pre_sigma[val_dateindex].values.reshape((-1, dim, n_add))
                                ), axis=-1),
                np.concatenate((ens_mu[val_dateindex].values.reshape((-1, dim, 1)),
                                ens_sigma[val_dateindex].values.reshape((-1, dim, 1)),
                                ens_max[val_dateindex].values.reshape((-1, dim, 1)),
                                ens_min[val_dateindex].values.reshape((-1, dim, 1)),
                                ens_spread[val_dateindex].values.reshape((-1, dim, 1)),
                                addpre[val_dateindex].values.reshape((-1, dim, add_dim))
                                ), axis=-1)]   
x_test_m323 = [np.concatenate((ens_mu[test_dateindex].values.reshape((-1, dim, 1)),
                                add_pre_mu[test_dateindex].values.reshape((-1, dim, n_add))
                                ), axis=-1),
                np.concatenate((ens_sigma[test_dateindex].values.reshape((-1, dim, 1)),
                                ens_max[test_dateindex].values.reshape((-1, dim, 1)),
                                ens_min[test_dateindex].values.reshape((-1, dim, 1)),
                                ens_spread[test_dateindex].values.reshape((-1, dim, 1)),
                                add_pre_sigma[test_dateindex].values.reshape((-1, dim, n_add))
                                ), axis=-1),
                np.concatenate((ens_mu[test_dateindex].values.reshape((-1, dim, 1)),
                                ens_sigma[test_dateindex].values.reshape((-1, dim, 1)),
                                ens_max[test_dateindex].values.reshape((-1, dim, 1)),
                                ens_min[test_dateindex].values.reshape((-1, dim, 1)),
                                ens_spread[test_dateindex].values.reshape((-1, dim, 1)),
                                addpre[test_dateindex].values.reshape((-1, dim, add_dim))
                                ), axis=-1)]     

y_train = obser[train_dateindex].values.reshape((-1, dim, 1))
y_val = obser[val_dateindex].values.reshape((-1, dim, 1))
y_test = obser[test_dateindex].values.reshape((-1, dim, 1))

y_train_tmp = y_train
y_val_tmp = y_val
y_test_tmp = y_test

testy = data_obs[test_dateindex]


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


# model 2
# Function to create model, fit and train model, test model, and get the ES output.

def igep323_es(params):

    dim_out=DIM
    dim_in_mean = x_train_m323[0].shape[-1] 
    dim_in_std = x_train_m323[1].shape[-1]
    dim_in_features = x_train_m323[2].shape[-1]
    n_samples_train=N_SAMPLES_TRAIN
    
    train_x=x_train_m323
    train_y=y_train_tmp
    val_x=x_val_m323
    val_y=y_val_tmp
    test_x=x_test_m323
    test_y=testy
    # latent_dist = 'uniform'
    # learningrate = 0.01
    epochs=EPOCHS
                      
    latent_dist_params = (0.0, 1.0)
    
    # nodes = 25
    # layer_number = 4
    nodes_number = params['nodes']
    layer_number = params['layers']
    
    dim_latent = params['dim_latent']
    # batch_size = params['batch_size']
    batch_size = 54
    learningrate = params['learningrate']
    latent_dist = params['latent_dist']
    n_channel = params['n_channel']
    
    tfv.reset_default_graph()
    
    ### Inputs ###
    input_mean = keras.Input(shape=(dim_out, dim_in_mean), name = "input_mean")
    input_sd = keras.Input(shape=(dim_out, dim_in_std), name = "input_sd")
    input_all = keras.Input(shape=(dim_out, dim_in_features), name = "input_all")
    bs = K.shape(input_mean)[0]
    
    #####
#    x_mean = layers.Flatten()(input_mean)
#    
#    for l in range(layer_number):
#        x_mean = layers.Dense(nodes_number, use_bias=True, activation = 'elu')(x_mean)
#        
#    x_mean = layers.Dense(dim_out, use_bias=True, activation = 'linear')(x_mean) # (, dim_out*1)
#    x_mean = layers.Reshape((dim_out, 1))(x_mean) # (, dim_out, 1)
    #####
    x_mean = layers.LocallyConnected1D(filters=n_channel, 
                                       kernel_size=1, 
                                       strides=1,
                                       padding='valid',
                                       data_format='channels_last',
                                       use_bias=True,
                                       activation='linear')(input_mean) # (, dim_out, n)
    x_mean = layers.LocallyConnected1D(filters=1, 
                                       kernel_size=1, 
                                       strides=1,
                                       padding='valid',
                                       data_format='channels_last',
                                       use_bias=True,
                                       activation='linear')(x_mean) # (, dim_out, 1)
    #####  
    
    x_mean_all = layers.Lambda(lambda arg: K.repeat_elements(arg, n_samples_train, axis=-1))(x_mean) # (, dim_out, n_samples)
    
    #####
#    z_delta = layers.Flatten()(input_sd)
#    
#    for l in range(layer_number):
#        z_delta = layers.Dense(nodes_number, use_bias=True, activation = 'elu')(z_delta)
#    
#    z_delta = layers.Dense(dim_out, use_bias=True, activation = 'linear')(z_delta) # (, dim_out*1)
    #####
    z_delta = layers.LocallyConnected1D(filters=20, 
                                       kernel_size=1, 
                                       strides=1,
                                       padding='valid',
                                       data_format='channels_last',
                                       use_bias=True,
                                       activation='linear')(input_sd) # (, dim_out, 20)
    z_delta = layers.LocallyConnected1D(filters=1, 
                                       kernel_size=1, 
                                       strides=1,
                                       padding='valid',
                                       data_format='channels_last',
                                       use_bias=True,
                                       activation='linear')(z_delta) # (, dim_out, 1)
    #####        
    
    z_delta_flat = layers.Flatten()(z_delta)
    
    z_delta_final = layers.Dense(dim_latent, activation = 'exponential')(z_delta_flat) # spread of latent variables 
    
    z_delta_reshape = layers.Lambda(lambda arg: K.reshape(arg, (bs, dim_latent, 1)))(z_delta_final) # (, dim_latent, 1)
    
    if latent_dist == "uniform":
        z = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                 minval=latent_dist_params[0], 
                                                 maxval=latent_dist_params[1]))([bs, dim_latent, n_samples_train])
    elif latent_dist == "normal":
        z = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                mean=latent_dist_params[0], 
                                                stddev=latent_dist_params[1]))([bs, dim_latent, n_samples_train])
   
    z_adjust_spread = layers.Multiply()([z_delta_reshape, z]) # (, dim_latent, n_samples)
    
    # weights
    #####
    W = layers.Flatten()(input_all)
    for l in range(layer_number):
        W = layers.Dense(nodes_number, use_bias=True, activation = 'elu')(W)
    
    W = layers.Dense(dim_out*dim_latent, use_bias=True, activation = 'linear')(W) # (, dim_out*dim_latent)
    W = layers.Reshape((dim_out, dim_latent))(W) # (, dim_out, dim_latent)
    #####
    
    z_samples = layers.Dot(axes=(2,1))([W, z_adjust_spread]) # (, dim_out, n_samples)
    y = layers.Add()([x_mean_all, z_samples])
    
    
    model =  Model(inputs=[input_mean, input_sd, input_all], outputs=y)

    
    opt = keras.optimizers.Adam(learning_rate=learningrate) # lr default 0.01
    model.compile(loss=EnergyScore(), optimizer=opt)
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = 0.002, patience=3, restore_best_weights=True)
    model.fit(x=train_x, y=train_y,
              batch_size=batch_size, 
              epochs=epochs, 
              verbose=2, 
               callbacks=[callback], 
              validation_split=0.0, 
              validation_data=(val_x, val_y),
              shuffle=True,
              sample_weight=None)
    
    N_SAMPLES_TEST = 50
    S = []
    for i in range(np.int(np.ceil(N_SAMPLES_TEST/n_samples_train))):
        S.append(model.predict(test_x))
    predict_data = np.concatenate(S, axis=2)[:, :, 0:N_SAMPLES_TEST]
    
    P = []
    # P.append(scaler.inverse_transform(predict_data))
    P.append(predict_data)
    pre_dat = np.concatenate(P,axis=0)
    
    fcst = scaler.inverse_transform(np.reshape(pre_dat, (pre_dat.shape[0]*pre_dat.shape[1],-1)))
    
    ES = es_sample(y = np.reshape(test_y.values, (-1, DIM)), dat = np.reshape(fcst, 
    (pre_dat.shape[0],pre_dat.shape[1],pre_dat.shape[2]) ) )
    
    return {'loss': ES, 'status': STATUS_OK, 'model': model, 'params': params}



# quniform returns float, some parameters require int; use this to force int

space_m2 = {'dim_latent' : hp.uniformint('dim_latent',5,35),
            'layers' : hp.uniformint('layers', 1, 3),
            'nodes' : hp.uniformint('nodes', 10,40),
            'latent_dist' : hp.choice('latent_dist', ['normal', 'uniform']),
            'learningrate' : hp.choice('learningrate', [0.01, 0.001]),
            'n_channel' : hp.choice('n_channel', [4,8,10,16,20,32,64,128])}
# space_m2 = {'dim_latent' : hp.uniformint('dim_latent',30,40),
#          'batch_size' : hp.uniformint('batch_size',53,56)}


trials2 = Trials()
best_m2 = fmin(igep323_es, 
            space_m2, 
            algo=tpe.suggest,
            max_evals=42,
            trials=trials2)

trials_m2 = trials2.trials
results_m2 = trials2.results
es_m2 = trials2.losses()


for trial in trials_m2:
        trial['result'].pop('model', None) 

pickle.dump(trials_m2, open('trials_ig326.p', "wb"))

print(best_m2)




