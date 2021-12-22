
import numpy as np
import pandas as pd
# import xarray as xr
# import xskillscore

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.close("all")

from scoringRules import es_sample, crps_sample
from igep_models_all_tem_noplot import igep

import tensorflow.compat.v1 as tfv
tfv.disable_v2_behavior()



# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# OPTIONS
# set model hyper parameters
N_SAMPLES_TRAIN = 50     # number of samples drawn during training
# EPOCHS = 50       # max epochs      # number of epochs in model training #= 50

# misc options
VERBOSE = 1                     # determines how much info is given about fitting
N_SAMPLES_TEST = 10 # 1000           # number of samples used for computung scores
TAUS = np.linspace(1,99,99)/100 # which quantiles to evaluate
PLOT_RESULTS = False

# Read data
path = '/home/chen_jieyu/IGEP/ECMWF_wind_data.feather'
t2m_ens_complete = pd.read_feather(path)

path_add = '/home/chen_jieyu/IGEP/wind_additional_predictors.feather'
t2m_add_complete = pd.read_feather(path_add)

dist_samples = pd.read_csv('/home/chen_jieyu/IGEP/ws_dist_10samples.csv', header=None)
DIM = 10

callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.002, patience = 3, restore_best_weights = True)

# loop
x = 100

for k in range(x):
    
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
    
    
    # SPLIT DATA
    
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
                               "sshf_mean", "slhf_mean", "ssr_mean", "str_mean","sin_yday"]]
    
    add_pre_sigma = addpre.loc[:,["sp_var","u_pl850_var","v_pl850_var","ws_pl850_var","q_pl850_var",
                                  "u_pl500_var", "v_pl500_var", "ws_pl500_var", "u10_var", "v10_var",
                                  "t2m_var", "d2m_var", "cape_var", "tcc_var", "gh_pl500_var",
                                  "sshf_var", "slhf_var", "ssr_var", "str_var","sin_yday"]]
    
    n_add = 20
    
    # Inputs for Model 3.2.3
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
    
    
    # MODEL 323
    
    BATCH_SIZE = 54           # batch size in model training #= 28 / 30
    LATENT_DIST = "normal"   # or uniform # family of latent varaible distributions
    DIM_LATENT = 26           # number of latent variables
    learning_rate = 0.001
    EPOCHS = 50
    n_layers = 2
    n_nodes = 30
    ens_m3_output_combine = pd.DataFrame()
    
    for loop in range(5):
    
        tfv.reset_default_graph()
        
        # initialize model
        mdl_m3 = igep(dim_out=DIM, 
                      dim_in_mean = x_train_m323[0].shape[-1], 
                      dim_in_std = x_train_m323[1].shape[-1],
                      dim_in_features = x_train_m323[2].shape[-1], 
                      dim_latent=DIM_LATENT, 
                      n_samples_train=N_SAMPLES_TRAIN, 
                      layer_number=n_layers,
                      nodes_number=n_nodes,
                      model_type=325, 
                      latent_dist=LATENT_DIST)
        
        
        #% FIT
        mdl_m3.fit(x=x_train_m323, 
                    y=y_train_tmp, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, 
                    verbose=VERBOSE, 
                    callbacks=[callback],
                    validation_split=0.0,
                    validation_data=(x_val_m323, y_val_tmp),         
                    sample_weight=None,
                    learningrate = learning_rate)
        
        # predict and append to list
        S_m3 = []
        # S_m3.append(scaler.inverse_transform(mdl_m3.predict(x_test_m323, N_SAMPLES_TEST)))
        S_m3.append(mdl_m3.predict(x_test_m323, N_SAMPLES_TEST))
        S_m3 = np.concatenate(S_m3,axis=0)
    
        ens_m3_output = pd.DataFrame(np.reshape(S_m3, (S_m3.shape[0]*S_m3.shape[1],-1)), index=testy.index)
        
        ens_m3_output_combine = pd.concat([ens_m3_output_combine, ens_m3_output], axis=1)
    
    
    ens_m3_result = pd.concat([testy, ens_m3_output_combine], axis=1)
    
    file_name="ws_10dim_ig325_"+str(k)+".csv"
    ens_m3_result.to_csv('/home/chen_jieyu/IGEP/ig325_wind_fixhp/'+file_name)
    
    print('m3:'+str(k))
    
    



