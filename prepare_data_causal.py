import time

from sample_causal import sample_causal
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing
from joblib import Parallel, delayed

def prepare_data_causal(explainer, seed=None, n_samples = 1000, index_features = None, asymmetric = False, ordering = None, args=[]):
    # Don't know if something similar should be done in python? line below
    # id <- id_combination <- w <- NULL # due to NSE notes in R CMD check

    #Probably need a new explainer class here.
    n_xtest = len(explainer.x_test.index)  # Assume that x_test is a pandas DataFrame
    dt_l = []

    if seed is not None:
        np.random.seed(seed=seed)

    if index_features is None:
        features = explainer.X.features

    else:
        features = explainer.X.features[index_features]

    if asymmetric is True:
        #Could have a diagnostic message here instead
        exit("Asymmetric is not yet implemented")
        print("Asymmetric flag enabled. Only using permutations consistent with the ordering.")

        # By default, no ordering in specified, meaning all variables are in one component.
        if (ordering is None):
            print("Using no ordering by default.")
            ordering = [range(len(explainer.x_test.columns))]

        # Filter out the features that do not agree with the order
        # TODO, assuming that we won't use asymmetric at least in the beginning
        #  features = features[]
        #  features <- features[sapply(features, respects_order, ordering = ordering)]

    num_cores = multiprocessing.cpu_count()
    inputs = range(n_xtest)
    seeds = np.random.randint(2**32-1, size=n_xtest)
    dt_l = Parallel(n_jobs=num_cores)(delayed(sample_causal_loop)(i, n_samples, explainer,
                                                                  features, index_features, seeds[i]) for i in inputs)
    #print(len(sample_causal_list))
    #exit()
    """for i in range(len(dt_l)):
        print(dt_l[i]['id'])    
    exit()"""
    dt = pd.concat(dt_l, ignore_index=True)
    dt.loc[dt.id_combination == 0, 'w'] = 1.0
    dt.loc[dt.id_combination == 2**(len(explainer.x_test.columns))-1, 'w'] = 1.0
    return dt

def sample_causal_loop(i, n_samples, explainer, features, index_features, seed):
    # print(str(i)+" of "+str(n_xtest)+" n_xtest")
    l = [sample_causal(feature_element, n_samples, explainer.mu, explainer.cov_mat, len(explainer.x_test.columns),
                       explainer.x_test.iloc[i], ordering=explainer.ordering, confounding=explainer.confounding,
                       seed=seed) for feature_element in features]
    for j in range(len(l)):
        l[j].insert(0, 'id_combination', j)

    l = pd.concat(l, ignore_index=True)
    l['w'] = 1 / n_samples
    l['id'] = i

    if index_features is not None:
        #   don't know what is happening here
        exit("prepare_data_causal: index_features is not None")
    return l