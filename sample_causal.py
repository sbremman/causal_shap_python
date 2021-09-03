import numpy as np
import numpy.random
import pandas as pd
import time


# index_given: numpy.array (1d)
# n_samples: int
# mu: numpy.array (1d)
# cov_mat: numpy.array(2d)
# m: int (features?)
# x_test: pandas.DataFrame
# ordering: list
# confounding: list

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def symmetric_part(a):
    return (a + a.T) / 2

def sample_causal(index_given, n_samples, mu, cov_mat, m, x_test, ordering=None, confounding=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if ordering is None:
        ordering = [range(m)]

    if confounding is None:
        confounding = [False] * len(ordering)
    # TODO Check input
    # TODO Check is x_test is a matrix...
    # TODO Check if ordering is a list...

    # TODO Check if confounding is specified globally or separately for each causal component...

    # TODO If confounding specified globally, replicate value for each component

    # TODO Check if Incomplete or incorrect partial ordering is specified for , m, variables


    dependent_ind = np.setdiff1d(np.arange(len(mu)), index_given)
    xall = np.empty([n_samples, m])

    x_test_numpy = x_test.to_numpy()
    xall[:, index_given] = np.repeat(np.array([x_test_numpy[index_given]]), n_samples, axis=0)
    for i in range(len(ordering)):
        # check overlap between dependent_ind and component
        to_be_sampled = np.intersect1d(ordering[i], dependent_ind)
        if len(to_be_sampled) > 0:
            # condition upon all variables in ancestor components
            to_be_conditioned = [item for sublist in ordering[0:i] for item in sublist]

            # back to conditioning if confounding is FALSE or no conditioning if confounding is TRUE
            if not confounding[i]:
                # add intervened variables in the same component
                to_be_conditioned = np.union1d(to_be_conditioned, np.intersect1d(ordering[i], index_given)).astype(int)

            if len(to_be_conditioned) == 0:
                # Draw new samples from marginal distribution
                new_samples = np.random.multivariate_normal(mean=mu[to_be_sampled],
                                                            cov=cov_mat[np.ix_(to_be_sampled, to_be_sampled)],
                                                            size=n_samples)


            else:
                # Compute conditional Gaussian
                c, d = cov_mat[np.ix_(to_be_sampled, to_be_conditioned)], \
                       cov_mat[np.ix_(to_be_conditioned, to_be_conditioned)]
                cd_inv = c.dot(np.linalg.inv(d))
                cVar = cov_mat[np.ix_(to_be_sampled, to_be_sampled)] - cd_inv.dot(c.T)
                if not check_symmetric(cVar):
                    cVar = symmetric_part(cVar)

                # draw new samples from conditional distribution
                mu_sample, mu_cond = np.repeat(np.array([mu[to_be_sampled]]), n_samples, axis=0), \
                                     np.repeat(np.array([mu[to_be_conditioned]]), n_samples, axis=0)

                cMU, new_samples = mu_sample + cd_inv.dot((xall[:, to_be_conditioned] - mu_cond).T).T, \
                                   np.random.multivariate_normal(mean=np.array([0] * len(to_be_sampled)),
                                                                 cov=cVar,
                                                                 size=n_samples)
                new_samples = new_samples + cMU

            xall[:, to_be_sampled] = new_samples

    xall_df = pd.DataFrame(xall, columns=x_test.index.values)

    return xall_df