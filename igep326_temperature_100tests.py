
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



DIM = 5                        # dimension of target values
dist_samples = pd.read_csv('/home/chen_jieyu/IGEP/dist_5samples.csv', header = None)


# Read data
path = '/home/chen_jieyu/IGEP/ens_fc_t2m_complete.feather'
t2m_ens_complete = pd.read_feather(path)

path_add = '/home/chen_jieyu/IGEP/tem_additional_predictors.feather'
t2m_add_complete = pd.read_feather(path_add)


callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.002, patience = 3, restore_best_weights = True)

# loop
x = 100

for k in range(x):
    station_sample = dist_samples.iloc[k,]
    
    ens_sample = t2m_ens_complete[t2m_ens_complete['station'].isin(station_sample)]

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
    
    # Inputs
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
    
    
    # MODEL
    
    BATCH_SIZE = 64  
    LATENT_DIST = "uniform"   # or normal # family of latent varaible distributions
    DIM_LATENT = 20           # number of latent variables
    learning_rate = 0.01
    EPOCHS = 50
    N_SAMPLES_TRAIN = 50     # number of samples drawn during training
    N_SAMPLES_TEST = 100
    VERBOSE = 1
    n_layers = 2
    n_nodes = 25
    ens_m3_output_combine_l = pd.DataFrame()
    ens_m3_output_combine_s = pd.DataFrame()
    
    for loop in range(5):
    
        tfv.reset_default_graph()
        
        # initialize model
        mdl_m3 = igep(dim_out = DIM, 
                      dim_in_mean = x_train_m323[0].shape[-1], 
                      dim_in_std = x_train_m323[1].shape[-1],
                      dim_in_features = x_train_m323[2].shape[-1], 
                      dim_latent = DIM_LATENT, 
                      n_samples_train = N_SAMPLES_TRAIN, 
                      layer_number = n_layers,
                      nodes_number = n_nodes,
                      model_type = 326, 
                      latent_dist = LATENT_DIST)
        
        
        #% FIT
        mdl_m3.fit(x = x_train_m323, 
                    y = y_train_tmp, 
                    batch_size = BATCH_SIZE, 
                    epochs = EPOCHS, 
                    verbose = VERBOSE, 
                    callbacks = [callback],
                    validation_split = 0.0,
                    validation_data = (x_val_m323, y_val_tmp),         
                    sample_weight = None,
                    learningrate = learning_rate)
        
        # predict and append to list
        S_m3 = []
        S_m3.append(mdl_m3.predict(x_test_m323, N_SAMPLES_TEST))
        pre_dat = np.concatenate(S_m3, axis = 0)
    
        fcst = scaler.inverse_transform(np.reshape(pre_dat, (pre_dat.shape[0]*pre_dat.shape[1],-1)))
        
        ens_m3_output = pd.DataFrame(fcst, index=testy.index)
        
        ens_m3_output_combine_l = pd.concat([ens_m3_output_combine_l, ens_m3_output], axis=1)
        ens_m3_output_combine_s = pd.concat([ens_m3_output_combine_s, ens_m3_output.iloc[:, :10] ], axis=1)
    
    
    ens_m3_long_result = pd.concat([testy, ens_m3_output_combine_l], axis=1)
    ens_m3_short_result = pd.concat([testy, ens_m3_output_combine_s], axis=1)
        
    file_name_l = "ig326_t2m_5dim_long_" + str(k) + ".csv"
    ens_m3_long_result.to_csv('/Data/Jieyu_data/ig326_t2m_all/' + file_name_l)

    file_name_s = "ig326_t2m_5dim_short_" + str(k) + ".csv"
    ens_m3_short_result.to_csv('/Data/Jieyu_data/ig326_t2m_all/' + file_name_s)
    
    print('m3:'+str(k))
    
    



