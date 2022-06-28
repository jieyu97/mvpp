
import numpy as np
import pandas as pd

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
from cgm_models import cgm

import tensorflow.compat.v1 as tfv
tfv.disable_v2_behavior()


dim = 20                        # dimension of target values
var = 'ws'

dist_samples = pd.read_csv('/home/chen_jieyu/IGEP/'+ var +'_dist_'+ str(dim) +'samples.csv', header = None)


# Read data
path = '/home/chen_jieyu/IGEP/windspeed_data_cgm_std.feather'
data_complete = pd.read_feather(path)


callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.001, patience = 3, restore_best_weights = True)


# loop
n_rep = 100

for k in range(n_rep):
    
    station_sample = dist_samples.iloc[k,]
    
    data_subset = data_complete[data_complete['station'].isin(station_sample)]

    dateobs_count = data_subset.groupby('date')['date'].count()
    dates = dateobs_count.index
    used_dates = dates[dateobs_count == dim]
    used_data_subset = data_subset[data_subset['date'].isin(used_dates)]
    
    # Observation data
    data_obs = used_data_subset['obs']
    data_obs.index = used_data_subset['date']
    
    # set initial training and test dates
    train_dateindex = ((data_obs.index.year != 2016) & (data_obs.index.year != 2015))
    val_dateindex = (data_obs.index.year == 2015)
    test_dateindex = (data_obs.index.year == 2016)
    
    # Raw ensemble forecast data - 50 members
    data_ens = used_data_subset.iloc[:, 3:53]
    data_ens.index = used_data_subset['date']
    
    # All predictors data: summary statistics of variables, location & time information
    n_addpred = 97 - 55
    data_addpred = used_data_subset.iloc[:, 55:97]
    data_addpred.index = used_data_subset['date']

    
    # Split training, validation, and test set
    norm_obs = data_obs.copy()
    norm_ens = data_ens.copy()
    norm_addpred = data_addpred.copy()

    observation = data_obs.copy()
    ensemble = data_ens.copy()
    
    ens_mu = ensemble.mean(axis=1)
    ens_sigma = ensemble.std(axis=1)
    
    ######### Normalization    
    scaler = preprocessing.MinMaxScaler().fit(norm_obs[train_dateindex].values.reshape(-1,1)) 
    stand_obs = scaler.transform(norm_obs.values.reshape(-1,1)).reshape(-1)
    norm_obs.iloc[:] = stand_obs
    
    for i in range(norm_ens.shape[1]):
        norm_ens.iloc[:,i] = scaler.transform(norm_ens.iloc[:,i].values.reshape(-1,1))
    
    norm_ens_mu = norm_ens.mean(axis=1)
    norm_ens_sigma = norm_ens.std(axis=1)
    
    norm_ens_max = norm_ens.max(axis=1)
    norm_ens_min = norm_ens.min(axis=1)
    norm_ens_spread = norm_ens_max - norm_ens_min
    
    for i in range(norm_addpred.shape[1]-1):
        scaler_i = preprocessing.StandardScaler().fit(norm_addpred.iloc[train_dateindex,i].values.reshape(-1,1))
        norm_addpred.iloc[:,i] = scaler_i.transform(norm_addpred.iloc[:,i].values.reshape(-1,1))
    
    norm_addvar_mu = norm_addpred.iloc[:, range(0, 38, 2)]
    norm_addvar_sigma = norm_addpred.iloc[:, range(1, 38, 2)]
    
    n_add_variable = 19
    
    # Inputs
    x_training = [np.concatenate((ens_mu[train_dateindex].values.reshape((-1, dim, 1)),
                                norm_addvar_mu[train_dateindex].values.reshape((-1, dim, n_add_variable))
                                ), axis=-1),
                np.concatenate((ens_sigma[train_dateindex].values.reshape((-1, dim, 1)),
                                norm_addvar_sigma[train_dateindex].values.reshape((-1, dim, n_add_variable))
                                ), axis=-1),
                np.concatenate((norm_ens_mu[train_dateindex].values.reshape((-1, dim, 1)),
                                norm_ens_sigma[train_dateindex].values.reshape((-1, dim, 1)),
                                norm_ens_max[train_dateindex].values.reshape((-1, dim, 1)),
                                norm_ens_min[train_dateindex].values.reshape((-1, dim, 1)),
                                norm_ens_spread[train_dateindex].values.reshape((-1, dim, 1)),
                                norm_addpred[train_dateindex].values.reshape((-1, dim, n_addpred))
                                ), axis=-1)]    
    x_validation = [np.concatenate((ens_mu[val_dateindex].values.reshape((-1, dim, 1)),
                                  norm_addvar_mu[val_dateindex].values.reshape((-1, dim, n_add_variable))
                                    ), axis=-1),
                    np.concatenate((ens_sigma[val_dateindex].values.reshape((-1, dim, 1)),
                                    norm_addvar_sigma[val_dateindex].values.reshape((-1, dim, n_add_variable))
                                    ), axis=-1),
                    np.concatenate((norm_ens_mu[val_dateindex].values.reshape((-1, dim, 1)),
                                    norm_ens_sigma[val_dateindex].values.reshape((-1, dim, 1)),
                                    norm_ens_max[val_dateindex].values.reshape((-1, dim, 1)),
                                    norm_ens_min[val_dateindex].values.reshape((-1, dim, 1)),
                                    norm_ens_spread[val_dateindex].values.reshape((-1, dim, 1)),
                                    norm_addpred[val_dateindex].values.reshape((-1, dim, n_addpred))
                                    ), axis=-1)]   
    x_test = [np.concatenate((ens_mu[test_dateindex].values.reshape((-1, dim, 1)),
                            norm_addvar_mu[test_dateindex].values.reshape((-1, dim, n_add_variable))
                            ), axis=-1),
            np.concatenate((ens_sigma[test_dateindex].values.reshape((-1, dim, 1)),
                            norm_addvar_sigma[test_dateindex].values.reshape((-1, dim, n_add_variable))
                            ), axis=-1),
            np.concatenate((norm_ens_mu[test_dateindex].values.reshape((-1, dim, 1)),
                            norm_ens_sigma[test_dateindex].values.reshape((-1, dim, 1)),
                            norm_ens_max[test_dateindex].values.reshape((-1, dim, 1)),
                            norm_ens_min[test_dateindex].values.reshape((-1, dim, 1)),
                            norm_ens_spread[test_dateindex].values.reshape((-1, dim, 1)),
                            norm_addpred[test_dateindex].values.reshape((-1, dim, n_addpred))
                            ), axis=-1)]     
    
    y_training = observation[train_dateindex].values.reshape((-1, dim, 1))
    y_validation = observation[val_dateindex].values.reshape((-1, dim, 1))
    y_test = observation[test_dateindex].values.reshape((-1, dim, 1))
    
    testy = data_obs[test_dateindex]
    
    
    # Model
    BATCH_SIZE = 64  
    LATENT_DIST = "normal"   # or uniform # family of latent varaible distributions
    DIM_LATENT = 26           # number of latent variables
    LEARNING_RATE = 0.001
    EPOCHS = 300
    N_SAMPLES_TRAIN = 50     # number of samples drawn during training
    N_SAMPLES_TEST = 100
    VERBOSE = 2
    ens_m3_output_combine_l = pd.DataFrame()
    ens_m3_output_combine_s = pd.DataFrame()
    
    for loop in range(5):
    
        tfv.reset_default_graph()
        
        # initialize model
        cgm_init = cgm(dim_out = dim, 
                        dim_in_mean = x_training[0].shape[-1], 
                        dim_in_std = x_training[1].shape[-1],
                        dim_in_features = x_training[2].shape[-1], 
                        dim_latent = DIM_LATENT, 
                        n_samples_train = N_SAMPLES_TRAIN,
                        model_type = var, 
                        latent_dist = LATENT_DIST)
        
        # fit model
        cgm_init.fit(x = x_training, 
                    y = y_training, 
                    batch_size = BATCH_SIZE, 
                    epochs = EPOCHS, 
                    verbose = VERBOSE, 
                    callbacks = [callback],
                    validation_split = 0.0,
                    validation_data = (x_validation, y_validation),         
                    sample_weight = None,
                    learningrate = LEARNING_RATE)
        
        # predict and append to list
        predictions = []
        predictions.append(cgm_init.predict(x_test, N_SAMPLES_TEST))
        pre_data = np.concatenate(predictions, axis = 0)
    
        ens_m3_output = pd.DataFrame(np.reshape(pre_data, (pre_data.shape[0]*pre_data.shape[1],-1)), index=testy.index)
       
        ens_m3_output_combine_l = pd.concat([ens_m3_output_combine_l, ens_m3_output], axis=1)
        ens_m3_output_combine_s = pd.concat([ens_m3_output_combine_s, ens_m3_output.iloc[:, :10] ], axis=1)
    
    
    ens_m3_long_result = pd.concat([testy, ens_m3_output_combine_l], axis=1)
    ens_m3_short_result = pd.concat([testy, ens_m3_output_combine_s], axis=1)
        
    file_name_l = var + "_" + str(dim) + "dim_long_sa" + str(k+1) + ".csv"
    ens_m3_long_result.to_csv('/Data/Jieyu_data/mvpp_cgm/' + file_name_l)

    file_name_s = var + "_" + str(dim) + "dim_sa" + str(k+1) + ".csv"
    ens_m3_short_result.to_csv('/Data/Jieyu_data/mvpp_cgm/' + file_name_s)
    
    print('CGM finished, round '+ str(k))
    
    



