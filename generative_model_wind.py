"""
@author: Jieyu Chen, Chair of statistics and econometrics @ Karlsruhe Institute of Technology, Germany

Tests of the conditional generative model for multivariate ensemble post-processing
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
import seaborn as sns

import tensorflow.compat.v1 as tfv
tfv.disable_v2_behavior()
sns.set()

from igep_models_all_wind_noplot import igep


DIM = 5     # dimension of target multivariate forecasts

# read 100 test subsets, each subset has DIM station
dist_samples = pd.read_csv('/home/chen_jieyu/IGEP/ws_dist_5samples.csv', header=None)

# read forecast and observation data of the target weather variable (10-m wind speed) and additional weather variables
path_add = '~/windspeed_data_cgm.feather'
ws_complete = pd.read_feather(path_add)

# set up early stoppong criterion
callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.002, patience = 3, restore_best_weights = True)

# repeat 100 tests on different subsets of stations
x = 100     # number of tests

for k in range(x):
    # select a subset of stations
    station_sample = dist_samples.iloc[k,]
    
    # extract the data at the test subset of stations
    ws_test_stations = ws_complete[ws_complete['station'].isin(station_sample)]
    
    # select days that have available forecasts & observations at all stations in the subset
    dateobs_count = ws_test_stations.groupby('date')['date'].count()
    dates = dateobs_count.index
    used_dates = dates[dateobs_count == DIM]
    
    # extract the data of the target variable on the selected days
    used_stations_sample = ws_test_stations[ws_test_stations['date'].isin(used_dates)]
    
    # extract observation data of the target variable
    ws_obs = used_stations_sample['obs']
    ws_obs.index = used_stations_sample['date']
    data_obs = ws_obs
        
    # extract ensemble forecast data of the target variable
    ws_ens = used_stations_sample.iloc[:,3:53]
    ws_ens.index = used_stations_sample['date']
    data_ens = ws_ens
    
    # split days into training, validation, and test sets acccording to the year
    train_dateindex = ((ws_obs.index.year != 2016) & (ws_obs.index.year != 2015))
    val_dateindex = (ws_obs.index.year == 2015)
    test_dateindex = (ws_obs.index.year == 2016)

    # select additional predictors: mean and variance of other weather variables, location and date information
    add_dim = 42     # number of additional predictors
    ws_add = used_stations_sample.loc[:,["sp_mean","sp_var",
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
    ws_add.index = used_stations_sample['date']
    data_add = ws_add

    # observations & raw ensemble forecasts of the target variable, additional predictors
    obser = data_obs.copy()
    pred = data_ens.copy()
    addpre = data_add.copy()
    dim = DIM

    ### standardization
    norm_obs = obser.copy()
    norm_pred = pred.copy()
    
    # the same standardization scaler for observations and raw ensemble forecasts
    scaler = preprocessing.MinMaxScaler().fit(norm_obs[train_dateindex].values.reshape(-1,1)) 
    stand_obs = scaler.transform(norm_obs.values.reshape(-1,1)).reshape(-1)
    norm_obs.iloc[:] = stand_obs
    
    for i in range(norm_pred.shape[1]):
        norm_pred.iloc[:,i] = scaler.transform(norm_pred.iloc[:,i].values.reshape(-1,1))
    
    # take the mean, variance, maximum value, minimum value, spread range of standardized ensembles
    norm_ens_mu = norm_pred.mean(axis=1)
    norm_ens_sigma = norm_pred.var(axis=1)
    
    norm_ens_max = norm_pred.max(axis=1)
    norm_ens_min = norm_pred.min(axis=1)
    norm_ens_spread = norm_ens_max - norm_ens_min
    
    # take the mean and variance of raw ensembles
    ens_mu = pred.mean(axis=1)
    ens_sigma = pred.var(axis=1)
    
    # using unique standardization scaler for each additional predictor
    for i in range(addpre.shape[1]-1):
        scaler_i = preprocessing.StandardScaler().fit(addpre.iloc[train_dateindex,i].values.reshape(-1,1))
        addpre.iloc[:,i] = scaler_i.transform(addpre.iloc[:,i].values.reshape(-1,1))
    
    # select mean predictor of additional weather variables and the date predictor "sin_yday"
    add_pre_mu = addpre.loc[:,["sp_mean", "u_pl850_mean", "v_pl850_mean", "ws_pl850_mean", "q_pl850_mean",
                               "u_pl500_mean", "v_pl500_mean", "ws_pl500_mean", "u10_mean", "v10_mean",
                               "t2m_mean", "d2m_mean", "cape_mean", "tcc_mean", "gh_pl500_mean",
                               "sshf_mean", "slhf_mean", "ssr_mean", "str_mean","sin_yday"]]
    
    # select variance predictor of additional weather variables and the date predictor "sin_yday"
    add_pre_sigma = addpre.loc[:,["sp_var","u_pl850_var","v_pl850_var","ws_pl850_var","q_pl850_var",
                                  "u_pl500_var", "v_pl500_var", "ws_pl500_var", "u10_var", "v10_var",
                                  "t2m_var", "d2m_var", "cape_var", "tcc_var", "gh_pl500_var",
                                  "sshf_var", "slhf_var", "ssr_var", "str_var","sin_yday"]]
    
    n_add = 20     # number of additional weather variables
    
    ### Inputs for the conditional generative models
    # three parts of inputs: mean of all weather variables; variance of all weather variables; all predictors
    # training set
    x_train_cgm = [np.concatenate((ens_mu[train_dateindex].values.reshape((-1, dim, 1)),
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
    # validation set
    x_val_cgm = [np.concatenate((ens_mu[val_dateindex].values.reshape((-1, dim, 1)),
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
    # test set
    x_test_cgm = [np.concatenate((ens_mu[test_dateindex].values.reshape((-1, dim, 1)),
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
    
    # labels of inputs (true observations) for computing loss
    y_train = obser[train_dateindex].values.reshape((-1, dim, 1))
    y_val = obser[val_dateindex].values.reshape((-1, dim, 1))
    y_test = obser[test_dateindex].values.reshape((-1, dim, 1))
    
    y_train_cgm = y_train
    y_val_cgm = y_val
    y_test_cgm = y_test
    
    testy = data_obs[test_dateindex]
    
    ### Conditional generative model
    
    # hyperparameter choices
    BATCH_SIZE = 64     # batch size
    LATENT_DIST = "normal"     # type of latent distribution
    DIM_LATENT = 26     # number of latent variables
    learning_rate = 0.001     # learning rate
    EPOCHS = 50     # maximum number of epochs
    N_SAMPLES_TRAIN = 50     # number of samples drawn during training to compute loss
    N_SAMPLES_TEST = 100     # number of samples drawn for the output
    VERBOSE = 1
    n_layers = 0
    n_nodes = 0
    ens_m3_output_combine_l = pd.DataFrame()
    ens_m3_output_combine_s = pd.DataFrame()
    
    # repeat the training for 5 times to reduce uncertainty
    for loop in range(5):
        
        # clean graph memory to save computational costs
        tfv.reset_default_graph()
        
        # initialize model
        mdl_m3 = igep(dim_out = DIM, 
                      dim_in_mean = x_train_cgm[0].shape[-1], 
                      dim_in_std = x_train_cgm[1].shape[-1],
                      dim_in_features = x_train_cgm[2].shape[-1], 
                      dim_latent = DIM_LATENT, 
                      n_samples_train = N_SAMPLES_TRAIN, 
                      layer_number = n_layers,
                      nodes_number = n_nodes,
                      model_type = 327, 
                      latent_dist = LATENT_DIST)
        
        # training model
        mdl_m3.fit(x = x_train_cgm, 
                    y = y_train_cgm, 
                    batch_size = BATCH_SIZE, 
                    epochs = EPOCHS, 
                    verbose = VERBOSE, 
                    callbacks = [callback],
                    validation_split = 0.0,
                    validation_data = (x_val_cgm, y_val_cgm),         
                    sample_weight = None,
                    learningrate = learning_rate)
        
        # predict and combine outputs
        S_m3 = []
        
        S_m3.append(mdl_m3.predict(x_test_cgm, N_SAMPLES_TEST))
        S_m3 = np.concatenate(S_m3,axis=0)
    
        ens_m3_output = pd.DataFrame(np.reshape(S_m3, (S_m3.shape[0]*S_m3.shape[1],-1)), index=testy.index)
        
        ens_m3_output_combine_l = pd.concat([ens_m3_output_combine_l, ens_m3_output], axis=1) # CGM with 500 samples
        ens_m3_output_combine_s = pd.concat([ens_m3_output_combine_s, ens_m3_output.iloc[:, :10] ], axis=1) # CGM with 50 samples
    
    # save post-processed forecasts
    ens_m3_long_result = pd.concat([testy, ens_m3_output_combine_l], axis=1)
    ens_m3_short_result = pd.concat([testy, ens_m3_output_combine_s], axis=1)
        
    file_name_l = "cgm_ws_5dim_long_" + str(k) + ".csv"
    ens_m3_long_result.to_csv('/Data/cgm_ws_all/' + file_name_l)

    file_name_s = "cgm_ws_5dim_short_" + str(k) + ".csv"
    ens_m3_short_result.to_csv('/Data/cgm_ws_all/' + file_name_s)
        
    print('m3:'+str(k))
    
    



