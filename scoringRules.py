"""
@author: Tim Janke, Energy Information Networks & Systems Lab @ TU Darmstadt, Germany

Mostly wrapper functions for the R scoringRules package using rpy2.
See https://cran.r-project.org/web/packages/scoringRules/ for details.
"""

import numpy as np
import rpy2.robjects as R
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri


# install scoring rules if necessary:
#if rpackages.isinstalled("scoringRules") == False:
#    r_utils = rpackages.importr('utils')
#    r_utils.chooseCRANmirror(ind=1) # select the first mirror in the list
#    r_utils.install_packages(R.vectors.StrVector(["scoringRules"])) #install scoringRules
    
    
scoringRules = rpackages.importr('scoringRules', lib_loc = '/home/chen_jieyu/R/x86_64-pc-linux-gnu-library/4.1/') # import scoringRules package
numpy2ri.activate()


# helper function to convert Python types to R types
def convert_to_Rtype(x):
    if x is None:
        x = R.NULL    
    elif type(x) is str:
        pass
    elif type(x) is int:
        pass
    elif type(x) is float:
        pass
    elif type(x) is np.ndarray:
        if x.ndim == 1:
            x = R.vectors.FloatVector(x)
        elif x.ndim == 2:
            x = R.r.matrix(x, nrow=x.shape[0], ncol=x.shape[1])
        else:
            raise TypeError("Input must be one of following: None, str, int, float, (n,) numpy array, (n,m) numpy array.")
    return x


########## univariate scores ###########
# Pinball Score
def pinball_score(y, dat, taus, return_single_scores=False, return_qloss=False):

    """
    Compute average Pinball Score from quantiles of the predictive distribution.

    Parameters
    ----------
    y : array, shape (N,)
        True values.
    dat : array, shape (N,n_taus)
        Predicted quantiles.
    taus : array, shape (n_taus,)
        Quantiles to evaluate.
    return_single_scores : bool, optional
        Return score for single examples. The default is False.
    return_qloss : bool, optional
        Return average scores for single quantiles. The default is False.

    Returns
    -------
    float or tuple.
        Default is to return mean pinball score.
        If return_single_scores is True, also returns array of single scores of shape (N,).
        If return_qloss is True, also returns array of average pinball scores per quantile of shape (n_taus,).

    """
    
    if len(y.shape) == 1:
        y = np.expand_dims(y,1)
    err = y-dat
    q_loss = np.maximum(err*taus,err*(taus-1))
    
    if return_single_scores is True and return_qloss is False:
        return (np.mean(q_loss), np.mean(q_loss,axis=1)) 
    elif return_single_scores is False and return_qloss is True:
        return (np.mean(q_loss), np.mean(q_loss,axis=0))
    elif return_single_scores is True and return_qloss is True: 
        return (np.mean(q_loss), np.mean(q_loss,axis=1), np.mean(q_loss,axis=0))
    else:
        return np.mean(q_loss)


# Continuous Ranked Probability Score (CRPS)
def crps_sample(y, dat, w=None, return_single_scores=False):
    """
    Compute Continuous Ranked Probability Score (CRPS) from samples of the predictive distribution.

    Parameters
    ----------
    y : array, shape(n_examples,)
        True values.
    dat : array, shape (n_examples, n_samples)
        Predictive scenarios.
    w : array matching shape of dat, optional
        Array of weights. The default is None.

    Returns
    -------
    float or tuple of (float, array)
        Returns average CRPS.
        If return_single_scores is True also returns array of scores for single examples.

    """
    scores = scoringRules.crps_sample(y=convert_to_Rtype(y),
                                     dat=convert_to_Rtype(dat),
                                     method="edf",
                                     w=convert_to_Rtype(w))
    if return_single_scores:
        return np.mean(scores), np.asarray(scores)
    else:
        return np.mean(scores)        

# Dawid-Sebastiani Score
def dss_sample(y, dat, w=None, return_single_scores=False):
    """
    Compute Dawid-Sebastiani Score (DSS) from samples of the predictive distribution.

    Parameters
    ----------
    y : array, shape(n_examples,)
        True values.
    dat : array, shape (n_examples, n_samples)
        Predictive scenarios.
    w : array matching shape of dat, optional
        Array of weights. The default is None.

    Returns
    -------
    float or tuple of (float, array)
        Returns average DSS.
        If return_single_scores is True, also returns array of scores for single examples.

    """
    scores = scoringRules.dss_sample(y=convert_to_Rtype(y),
                                     dat=convert_to_Rtype(dat),
                                     w=convert_to_Rtype(w))
    if return_single_scores:
        return np.mean(scores), np.asarray(scores)
    else:
        return np.mean(scores)  


# Log Score
def logs_sample(y, dat, bw=None, return_single_scores=False):
    """
    Compute log-Score from samples of the predictive distribution.

    Parameters
    ----------
    y : array, shape(n_examples,)
        True values.
    dat : array, shape (n_examples, n_samples)
        Predictive scenarios.
    bw : array matching shape of y, optional
        Array of bandwiths for KDE. The default is None.

    Returns
    -------
    float or tuple of (float, array)
        Returns average log-Score.
        If return_single_scores is True, also returns array of scores for single examples.

    """
    scores = scoringRules.logs_sample(y=convert_to_Rtype(y),
                                     dat=convert_to_Rtype(dat),
                                     bw=convert_to_Rtype(bw))
    if return_single_scores:
        return np.mean(scores), np.asarray(scores)
    else:
        return np.mean(scores)


########## multivariate scores ###########

# Energy Score
def es_sample(y, dat, return_single_scores=False):
    """
    Compute mean energy score from samples of the predictive distribution.

    Parameters
    ----------
    y : array, shape (n_examples, n_dim)
        True values.
    dat : array, shape (n_examples, n_dim, n_samples)
        Samples from predictive distribution.
    return_single_scores : bool, optional
        Return score for single examples. The default is False.

    Returns
    -------
    float or tuple of (float, array)
        Mean energy score. If return_single_scores is True also returns scores for single examples.

    """
    assert y.shape[0] == dat.shape[0], "y and dat must contain same number of examples."
    assert y.shape[1] == dat.shape[1], "Examples in y and dat must have same dimension."

    scores = []
    for i in range(dat.shape[0]):
        scores.append(scoringRules.es_sample(y=convert_to_Rtype(y[i,:]), 
                                             dat=convert_to_Rtype(dat[i,:,:])))
    if return_single_scores:
        return np.mean(scores), np.asarray(scores)
    else:
        return np.mean(scores)


# Variogram Score   
def vs_sample(y, dat, w=None, p=0.5, return_single_scores=False):
    """
    Compute mean variogram score from samples of the predictive distribution. 
    
    Parameters
    ----------
    y : array, shape (n_examples, n_dim)
        True values.
    dat : array, shape (n_examples, n_dim, n_samples)
        Samples from predictive distribution.
    w : array, shape (n_examples, n_dim, n_samples), optional
        Weights for observations and dimensions for dat. Must be postive. The default is None.
    p : float, optional
        Order of variagram score. The default is 0.5.
    return_single_scores : bool, optional
        Return score for single examples. The default is False.
    
    Returns
    -------
    float or tuple of (float, array)
        Average variogram score. If return_single_scores is True also returns scores for single examples.

    """
    assert y.shape[0] == dat.shape[0], "y and dat must contain same number of examples."
    assert y.shape[1] == dat.shape[1], "Examples in y and dat must have same dimension."
    
    scores = []
    for i in range(dat.shape[0]):
        if w is None:
            w_i = None
        else:
            w_i = w[i,:,:]
        scores.append(scoringRules.vs_sample(y=convert_to_Rtype(y[i,:]), 
                                             dat=convert_to_Rtype(dat[i,:,:]),
                                             w=convert_to_Rtype(w_i),
                                             p=p))
    if return_single_scores:
        return np.mean(scores), np.asarray(scores)
    else:
        return np.mean(scores)