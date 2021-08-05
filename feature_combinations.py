import sys
import pandas as pd
from itertools import chain, combinations
import scipy.special
import random
import numpy as np

def weight_matrix(X, normalize_W_weights = True):
    # Fetch weights
    w = X['shapley_weight'].copy()
    if normalize_W_weights:
        w[1:len(w)-1] = w[1:len(w)-1]/sum(w[1:len(w)-1])

    W = weight_matrix_temp(features=X['features'].to_numpy(), m=X['n_features'].iloc[-1], n=len(X.index), w=w.to_numpy())
    return W

def weight_matrix_temp(features, m, n, w):
    Z = np.zeros([n, m+1])
    X = np.zeros([n, m+1])

    for i in range(n):
        Z[i][0] = 1

        feature_vec = features[i]
        n_features = len(feature_vec)
        if n_features > 0:
            for j in range(n_features):
                Z[i][feature_vec[j]+1] = 1

    for i in range(n):

        for j in range(Z.shape[1]):
            X[i][j] = w[i] * Z[i][j]

    R = np.matmul(np.linalg.inv(np.matmul(X.T, Z)), X.T)
    return R

def helper_feature(m, feature_sample):
    x = feature_matrix(feature_sample, m)
    dt = pd.DataFrame(x)
    sample_frequency_dict = dt.value_counts().to_dict()
    sample_frequency = []
    for sample in x:
        sample_frequency.append(sample_frequency_dict[tuple(sample)])
    dt["sample_frequency"] = sample_frequency
    dt["duplicated"] = dt.duplicated()
    dt = dt.drop(range(10), axis=1)
    return dt


def feature_matrix(feature_sample, m):
    feature_mat = []
    for features in feature_sample:
        feature_vector = np.array([0] * m)
        feature_vector[features] = 1
        feature_mat.append(feature_vector.tolist())

    return feature_mat


def shapley_weights(m, N, n_features, weight_zero_m=10 ** 6):
    # TODO Maybe return value should be rounded
    if N * n_features * (m - n_features) == 0:
        return weight_zero_m
    else:
        return (m - 1) / (N * n_features * (m - n_features))


def powerset(iterable):
    """" Taken from https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


def feature_combinations(m, exact=True, n_combinations=200, weight_zero_m=10 ** 6, asymmetric=False, ordering=None):
    if m > 12 and n_combinations is None:
        raise Exception("Due to computational complexity, we recommend setting n_combinations = 10 000\n if the number of "
                 "features is larger than 12. Note that you can force the use of the exact\n"
                 "method (i.e. n_combinations = NULL) by setting n_combinations equal to 2^m,\n"
                 "where m is the number of features."
                 )

    if m > 30:
        sys.exit("Currently we are not supporting cases where the number of features is greater than 30.")

    if not exact and n_combinations > 2 ** m - 2:
        n_combinations = 2 ** m - 2
        exact = True
        print("n_combinations is larger than or equal to 2^m = %d" % 2 ** m)

    if exact:
        dt = feature_exact(m, weight_zero_m, asymmetric, ordering)

    else:
        dt = feature_not_exact(m, n_combinations, weight_zero_m)
        if not isinstance(dt, pd.DataFrame):
            sys.exit("dt should be a pandas dataframe")

        if not 'p' in dt:
            sys.exit("No p in dt. Why this matters, I do not know... It is deleted right after anyway...")

        del dt['p']

    return dt


def feature_exact(m, weight_zero_m=10 ** 6, asymmetric=False, ordering=None):
    dt = pd.DataFrame(range(2 ** m), columns=['id_combinations'])
    combinations = powerset(range(0, m))
    combinations = [list(entry) for entry in combinations]
    dt['features'] = combinations
    dt['n_features'] = [len(combination) for combination in combinations]
    value_counts = dt['n_features'].value_counts(dropna=True).to_dict()
    N = []
    for item in dt['n_features']:
        N.append(value_counts[item])
    dt['N'] = N
    dt['shapley_weight'] = [shapley_weights(m, dt['N'][i], dt['n_features'][i], weight_zero_m) for i in
                            range(len(dt.index))]

    # Assume again that we won't use asymmetric
    if asymmetric:
        sys.exit("Asymmetric is not implemented")
        # if ordering is None:
        #     print("feature_exact: using no ordering by default")
        #     ordering = range(m)

    return dt


def feature_not_exact(m, n_combinations=200, weight_zero_m=10 ** 6):
    # Find weights for given number of features
    n_features = range(1, m)
    n = [scipy.special.comb(m, n_element, exact=True) for n_element in n_features]
    w = [shapley_weights(m, n[i], n_features[i]) * n[i] for i in range(len(n))]
    p = [x / sum(w) for x in w]

    # Sample number of chosen features
    X_list = random.choices(n_features, p, k=n_combinations)
    X_list.insert(0, 0)
    X_list.append(m)
    X = pd.DataFrame(X_list, columns=['n_features'])

    # Sample specific set of features
    X = X.sort_values(by=['n_features'], ignore_index=True)
    feature_sample = []
    # TODO This is written in c++ for shapr, probably much faster, should look into that
    for value in X['n_features']:
        # feature_sample.append(random.choices(n_features, k=value))
        feature_sample.append(sorted(random.sample(range(m), value)))

    # Get number of occurrences and duplicated rows
    r = helper_feature(m, feature_sample)
    X['is_duplicate'] = r['duplicated']

    # When we sample combinations the Shapley weight is equal
    # to the frequency of the given combination
    X['shapley_weight'] = r['sample_frequency']

    # Populate table and remove duplicated rows
    X['features'] = feature_sample

    if True in X['is_duplicate'].values:
        X = X[X.is_duplicate == False]

    del X['is_duplicate']

    # Add shapley weight and number of combinations
    X['shapley_weight'].iloc[[0, -1]] = weight_zero_m
    X['N'] = 1

    # TODO does not seem like r code uses p at all? It just removes it right after... Therefore I don't include it

    X = X.reset_index(drop=True)
    N = []
    P = []
    for item in X['n_features']:
        if item == 0 or item == 10:
            N.append(1)
            P.append(None)
        else:
            N.append(n[item-1])
            P.append(p[item-1])
    X['N'] = N
    X['p'] = P

    # Set column order
    X['id_combination'] = range(len(X))
    X = X[['id_combination', 'features', 'n_features', 'N', 'shapley_weight', 'p']]

    return X
