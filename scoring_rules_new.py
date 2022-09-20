"""
@author: Tim Janke

"""

import numpy as np
from scipy.stats import rankdata

########## multivariate scores ###########


# Energy Scores
def es_sample(y_true, y_pred, return_single_scores=False):
    """
    Compute mean energy score from samples of the predictive distribution.

    Parameters
    ----------
    y_true : array, shape (n_examples, n_dim)
        True values.
    y_pred : array, shape (n_examples, n_dim, n_samples)
        Samples from predictive distribution.
    return_single_scores : bool, optional
        Return score for single examples. The default is False.

    Returns
    -------
    float or tuple of (float, array)
        Mean energy score. If return_single_scores is True also returns scores for single examples.

    """
    assert len(y_pred.shape) == 3, "y_pred must be a three dimesnional array of shape (n_examples, n_dim, n_samples)"
    assert len(y_true.shape) == 2, "y_true must be a two dimesnional array of shape (n_examples, n_dim)"

    assert y_true.shape[0] == y_pred.shape[0], "y_true and y_pred must contain same number of examples."
    assert y_true.shape[1] == y_pred.shape[1], "Examples in y_true and y_pred must have same dimension."


    N = y_true.shape[0]
    M = y_pred.shape[2]

    es_12 = np.zeros(y_true.shape[0])
    es_22 = np.zeros(y_true.shape[0])

    for i in range(N):
        es_12[i] = np.sum(np.sqrt(np.sum(np.square((y_true[[i],:].T - y_pred[i,:,:])), axis=0)))
        es_22[i] = np.sum(np.sqrt(np.sum(np.square(np.expand_dims(y_pred[i,:,:], axis=2) - np.expand_dims(y_pred[i,:,:], axis=1)), axis=0)))
    
    scores = es_12/M - 0.5* 1/(M*M) * es_22
    if return_single_scores:
        return np.mean(scores), scores
    else:
        return np.mean(scores)


def ces_sample(y_true, y_pred, return_single_scores=False):
    """
    Compute mean copula energy score from samples of the predictive distribution.
    See Ziel & Berk(2019): "Multivariate Forecasting Evaluation: On Sensitive and Strictly Proper Scoring Rules" (https://arxiv.org/abs/1910.07325)

    Parameters
    ----------
    y_true : array, shape (n_examples, n_dim)
        True values.
    y_pred : array, shape (n_examples, n_dim, n_samples)
        Samples from predictive distribution.
    return_single_scores : bool, optional
        Return score for single examples. The default is False.

    Returns
    -------
    float or tuple of (float, array)
        Mean copula energy score. If return_single_scores is True also returns scores for single examples.

    """
    assert len(y_pred.shape) == 3, "y_pred must be a three dimesnional array of shape (n_examples, n_dim, n_samples)"
    assert len(y_true.shape) == 2, "y_true must be a two dimesnional array of shape (n_examples, n_dim)"

    assert y_true.shape[0] == y_pred.shape[0], "y_true and y_pred must contain same number of examples."
    assert y_true.shape[1] == y_pred.shape[1], "Examples in y_true and y_pred must have same dimension."

    y_true_pobs, y_predict_pobs  = _get_pobs(y_true, y_pred)

    return es_sample(y_true_pobs, y_predict_pobs, return_single_scores=return_single_scores)


# Variogram Scores
def vs_sample(y_true, y_pred, p=0.5, return_single_scores=False):
    """
    Compute mean variogram score from samples of the predictive distribution. 
    
    Parameters
    ----------
    y_true : array, shape (n_examples, n_dim)
        True values.
    y_pred : array, shape (n_examples, n_dim, n_samples)
        Samples from predictive distribution.
    p : float, optional
        Order of variagram score. The default is 0.5.
    return_single_scores : bool, optional
        Return score for single examples. The default is False.
    
    Returns
    -------
    float or tuple of (float, array)
        Average variogram score. If return_single_scores is True also returns scores for single examples.

    """
    assert len(y_pred.shape) == 3, "y_pred must be a three dimesnional array of shape (n_examples, n_dim, n_samples)"
    assert len(y_true.shape) == 2, "y_true must be a two dimesnional array of shape (n_examples, n_dim)"

    assert y_true.shape[0] == y_pred.shape[0], "y_true and y_pred must contain same number of examples."
    assert y_true.shape[1] == y_pred.shape[1], "Examples in y_true and y_pred must have same dimension."

    N = y_true.shape[0]
    D = y_true.shape[1]


    scores = np.zeros(y_true.shape[0])
    for i in range(N):
        
        vs_1 = np.power(np.abs(y_true[[i],:].T - y_true[[i],:]), p)
        vs_2 = np.mean(np.power(np.abs(np.repeat(y_pred[i,:,:], repeats=D, axis=0) - np.tile(y_pred[i,:,:], (D,1))), p), axis=1)
        scores[i] = np.sum(np.square(np.ravel(vs_1) - vs_2))
    
    if return_single_scores:
        return np.mean(scores), scores
    else:
        return np.mean(scores)


def cvs_sample(y_true, y_pred, p=0.5, return_single_scores=False):
    """
    Compute mean copula variogram score from samples of the predictive distribution. 
    See Ziel & Berk(2019): "Multivariate Forecasting Evaluation: On Sensitive and Strictly Proper Scoring Rules" (https://arxiv.org/abs/1910.07325)
    
    Parameters
    ----------
    y_true : array, shape (n_examples, n_dim)
        True values.
    y_pred : array, shape (n_examples, n_dim, n_samples)
        Samples from predictive distribution.
    p : float, optional
        Order of variagram score. The default is 0.5.
    return_single_scores : bool, optional
        Return score for single examples. The default is False.
    
    Returns
    -------
    float or tuple of (float, array)
        Average copula variogram score. If return_single_scores is True also returns scores for single examples.

    """
    assert len(y_pred.shape) == 3, "y_pred must be a three dimesnional array of shape (n_examples, n_dim, n_samples)"
    assert len(y_true.shape) == 2, "y_true must be a two dimesnional array of shape (n_examples, n_dim)"

    assert y_true.shape[0] == y_pred.shape[0], "y_true and y_pred must contain same number of examples."
    assert y_true.shape[1] == y_pred.shape[1], "Examples in y_true and y_pred must have same dimension."

    y_true_pobs, y_predict_pobs  = _get_pobs(y_true, y_pred)
    
    return vs_sample(y_true_pobs, y_predict_pobs, p=p, return_single_scores=return_single_scores)



def _get_pobs(y, dat):
    """ Obtain pseudo observations for dat as well as for y under the distribution represented by samples in dat[i,d,:]."""
    N,D,M = dat.shape
    
    # rank data in each sampled scenario
    pobs_dat = np.zeros_like(dat)
    for n in range(N):
        for d in range(D):
            pobs_dat[n,d,:] = (2*_rank_data_random_tiebreaker(dat[n,d,:])-1)/(2*M)
            
    idx = np.argmin(np.abs(np.expand_dims(y, axis=2) - dat), axis=2) # return index of nearest rank for observed y for each i and d
    pobs_y = np.squeeze(np.take_along_axis(pobs_dat, indices=np.expand_dims(idx, axis=2), axis=2)) # select nearest rank

    # ensure uniformity for uncalibrated predictive distributions
    pobs_y_adjusted = np.zeros_like(pobs_y)
    for d in range(D):
        pobs_y_adjusted[:,d] = (2*_rank_data_random_tiebreaker(pobs_y[:,d])-1)/(2*N)

    return pobs_y_adjusted, pobs_dat



def _rank_data_random_tiebreaker(a):
    """ Ranks data in 1d array using ordinal ranking (i.e. all ranks are assigned) and breaks ties at random """
    
    idx = np.random.permutation(len(a)) # indexes for random shuffle
    
    ranks_shuffled = rankdata(a[idx], method="ordinal") # compute ranks of shuffled index
    
    idx_r = np.zeros_like(idx) # indexes to invert permutation
    idx_r[idx] = np.arange(len(idx))
    
    ranks = ranks_shuffled[idx_r] # restore original order
    
    return ranks


