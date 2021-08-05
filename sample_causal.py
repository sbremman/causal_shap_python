import numpy as np
import pandas as pd
import time


#index_given: numpy.array (1d)
#n_samples: int
#mu: numpy.array (1d)
#cov_mat: numpy.array(2d)
#m: int (features?)
#x_test: pandas.DataFrame
#ordering: list
#confounding: list

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def symmetric_part(a):
    return (a+a.T)/2

def sample_causal(index_given, n_samples, mu, cov_mat, m, x_test, ordering=None, confounding=None):
    #TODO differences in this is most likely because the R implementation uses a specific seed
    start_time_causal = time.perf_counter()
    cov_mat_np = cov_mat.to_numpy()

    if ordering is None:
        ordering = [range(m)]

    if confounding is None:
        confounding = [False]*len(ordering)
    #Check input
    #Check is x_test is a matrix...
    #Check if ordering is a list...

    #Check if confounding is specified globally or separately for each causal component...

    #If confounding specified globally, replicate value for each component

    #Check if Incomplete or incorrect partial ordering is specified for , m, variables

    #Assumes that mu can only be vector, not matrix is this right?
    dependent_ind = np.setdiff1d(np.array(range(len(mu))), index_given)
    x_test_numpy = x_test.to_numpy()
    xall = np.empty([n_samples, m])

    #Can remove the line below when works
    xall[:] = np.nan
    xall[:, index_given] = np.repeat(np.array([x_test_numpy[index_given]]), n_samples, axis=0)
    for i in range(len(ordering)):
        #check overlap between dependent_ind and component
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
                #cov=cov_mat.iloc[to_be_sampled, to_be_sampled],
                new_samples = np.random.multivariate_normal(mean=mu[to_be_sampled],
                                                            cov=cov_mat_np[np.ix_(to_be_sampled, to_be_sampled)],
                                                            size=n_samples)


            else:
                start_time = time.perf_counter()
                # Compute conditional Gaussian
                # code originally has something like drop=FALSE in the c matrix here?
                #c = cov_mat[to_be_sampled, to_be_conditioned[0]:to_be_conditioned[-1]+1]
                #d = cov_mat[to_be_conditioned, to_be_conditioned[0]:to_be_conditioned[-1]+1]
                #cov_mat_np = cov_mat.to_numpy()
                #exit()
                #print(cov_mat_np[(to_be_sampled, to_be_conditioned)])
                c = cov_mat_np[np.ix_(to_be_sampled, to_be_conditioned)]
                #c = cov_mat.iloc[to_be_sampled, to_be_conditioned]

                #print(c_np.shape)
                #exit()
                d = cov_mat_np[np.ix_(to_be_conditioned, to_be_conditioned)]
                #d = cov_mat.iloc[to_be_conditioned, to_be_conditioned]
                #print(d.shape)
                #print(d_np.shape)
                #cd_inv_np = c_np.dot(np.linalg.inv(d_np))
                cd_inv = c.dot(np.linalg.inv(d))
                #cd_inv = cd_inv.to_numpy()
                #print(cd_inv.shape)
                #print(cd_inv_np)
                #exit()
                stop_time = time.perf_counter()
                #print(f"2 Sample causal took {stop_time - start_time:0.6f} seconds")
                start_time = time.perf_counter()

                #c = c.to_numpy()
                start_time1 = time.perf_counter()
                #cVar = cov_mat.iloc[to_be_sampled, to_be_sampled] - cd_inv.dot(c.T)
                cVar = cov_mat_np[np.ix_(to_be_sampled, to_be_sampled)] - cd_inv.dot(c.T)
                stop_time1 = time.perf_counter()
                #print(f"3_1 Sample causal took {stop_time1 - start_time1:0.6f} seconds")
                start_time1 = time.perf_counter()
                if not check_symmetric(cVar):
                    cVar = symmetric_part(cVar)
                stop_time1 = time.perf_counter()
                #print(f"3_2 Sample causal took {stop_time1 - start_time1:0.6f} seconds")
                # draw new samples from conditional distribution
                mu_sample = np.repeat(np.array([mu[to_be_sampled]]), n_samples, axis=0)
                mu_cond = np.repeat(np.array([mu[to_be_conditioned]]), n_samples, axis=0)
                cMU = mu_sample + cd_inv.dot((xall[:, to_be_conditioned] - mu_cond).T).T


                new_samples = np.random.multivariate_normal(mean=np.array([0]*len(to_be_sampled)),
                                                            cov=cVar,
                                                            size=n_samples)
                new_samples = new_samples+cMU
                stop_time = time.perf_counter()
                #print(f"3 Sample causal took {stop_time - start_time:0.6f} seconds")
                #exit()


            xall[:,to_be_sampled] = new_samples

    xall_df = pd.DataFrame(xall, columns=x_test.index.values)
    stop_time_causal = time.perf_counter()
    #print(f"Sample causal total took {stop_time_causal - start_time_causal:0.6f} seconds")

    return xall_df





