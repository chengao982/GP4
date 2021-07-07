import numpy as np
import func
from scipy.stats import norm

def calc_path_prob(path, mymap, T, samples=None, S=1000):    
    ''' 
    Evaluate the performance of a path in terms of its on-time-arrival probability.
    When model is log or bi, the probability can't be calculated analytically. Instead, it is computed using samples.
    In that case, a set of 'samples' can be passed into the function or the function will generate 'S' samples of the road network.
    '''
    if mymap.model == 'G':
        # path = np.flatnonzero(x)
        mu_sum = np.sum(mymap.mu[path])
        cov_sum = np.sum(mymap.cov[path, path])
        return norm.cdf(T, mu_sum, np.sqrt(cov_sum))
    else:
        if samples is None:
            samples = func.generate_samples(mymap, S)
        x = np.zeros(mymap.n_link)
        x[path] = 1
        return np.sum(np.where(np.dot(samples.T, x) <= T, 1, 0)) / samples.shape[1]


def calc_post_prob(path, mymap, T, N, S):
    ''' 
    Evaluate the performance of a path in terms of its posterior probability.
    It is the same as how GP4 is evaluated.
    This criteria is used to unify standard and make fair comparison.
    'N' samples are drawn for the first link in a path.
    The posterior distribution of the remaining road network is calculated for each N.
    The performance is the average on-time-arrival probability over N samples for the rest of the path.
    '''
    v_hat = 0

    x = np.zeros(mymap.n_link)
    x[path] = 1
    x = x.reshape(-1, 1)
    x = np.delete(x,path[0],0)

    rng = np.random.default_rng()

    if mymap.model == 'G':
        mu_sub, cov_sub, cov_con = func.update_param(mymap.mu, mymap.cov, path[0])
        path_con = [i if i < path[0] else i-1 for i in path[1:]]
        for i in range(N):
            sample = np.random.normal(mu_sub[2], np.sqrt(cov_sub[22]))
            T_temp = T - sample
            if T_temp > 0:
                mu_con = func.update_mu(mu_sub, cov_sub, sample)
                mu_sum = np.sum(mu_con[path_con])
                cov_sum = np.sum(cov_con[path_con, path_con])
                v_hat += norm.cdf(T_temp, mu_sum, np.sqrt(cov_sum))

    elif mymap.model == 'log':
        mu_sub, cov_sub, cov_con = func.update_param(mymap.mu, mymap.cov, path[0])
        for i in range(N):
            sample = np.random.normal(mu_sub[2], np.sqrt(cov_sub[22]))
            T_temp = T - np.exp(sample)
            if T_temp > 0:
                mu_con = func.update_mu(mu_sub, cov_sub, sample)
                samples = np.exp(rng.multivariate_normal(mu_con.reshape(-1), cov_con, S, method=mymap.decom))
                v_hat += np.sum(np.where(np.dot(samples, x) <= T_temp, 1, 0)) / samples.shape[1]

    elif mymap.model == 'bi':
        mu1_sub, cov1_sub, cov1_con = func.update_param(mymap.mu, mymap.cov, path[0])
        mu2_sub, cov2_sub, cov2_con = func.update_param(mymap.mu2, mymap.cov2, path[0])
        for i in range(N):
            sample = func.generate_biGP_samples(mymap.phi_bi, mu1_sub[2], mu2_sub[2], cov1_sub[22], cov2_sub[22], 1).item()
            T_temp = T - sample
            if T_temp > 0:
                mu1_con = func.update_mu(mu1_sub, cov1_sub, sample)
                mu2_con = func.update_mu(mu2_sub, cov2_sub, sample)
                samples = func.generate_biGP_samples(mymap.phi_bi, mu1_con, mu2_con, cov1_con, cov2_con, S, method=mymap.decom)
                v_hat += np.sum(np.where(np.dot(samples, x) <= T_temp, 1, 0)) / samples.shape[1]

    return v_hat / N