import numpy as np
import func

def calc_post_prob(path, mymap, T, N, S, model='G', decom_method='cholesky'):
    ''' 
    evaluate the performance of a path in terms of its posterior probability (the same way as how GP4 is evaluated).
    '''

    v_hat = 0

    x = np.zeros(mymap.n_link)
    x[path] = 1
    x = x.reshape(-1, 1)
    x = np.delete(x,path[0],0)

    rng = np.random.default_rng()

    if model == 'G':
        mu_sub, cov_sub, cov_con = func.update_param(mymap.mu, mymap.cov, path[0])

        for i in range(N):
            sample = np.random.normal(mu_sub[2], np.sqrt(cov_sub[22]))
            T_temp = T - sample
            if T_temp > 0:
                mu_con = func.update_mu(mu_sub, cov_sub, sample)
                samples = rng.multivariate_normal(mu_con.reshape(-1), cov_con, S, method=decom_method)
                v_hat += np.sum(np.where(np.dot(samples, x) - T_temp <= 0, 1, 0)) / S

    elif model == 'log':
        mu_sub, cov_sub, cov_con = func.update_param(mymap.mu, mymap.cov, path[0])
        for i in range(N):
            sample = np.random.normal(mu_sub[2], np.sqrt(cov_sub[22]))
            T_temp = T - np.exp(sample)
            if T_temp > 0:
                mu_con = func.update_mu(mu_sub, cov_sub, sample)
                samples = np.exp(rng.multivariate_normal(mu_con.reshape(-1), cov_con, S, method=decom_method))
                v_hat += np.sum(np.where(np.dot(samples, x) - T_temp <= 0, 1, 0)) / S

    elif model == 'bi':
        mu1_sub, cov1_sub, cov1_con = func.update_param(mymap.mu, mymap.cov, path[0])
        mu2_sub, cov2_sub, cov2_con = func.update_param(mymap.mu2, mymap.cov2, path[0])
        for i in range(N):
            sample = func.generate_biGP_samples(mymap.phi_bi, mu1_sub[2], mu2_sub[2], cov1_sub[22], cov2_sub[22], 1).item()
            T_temp = T - sample
            if T_temp > 0:
                mu1_con = func.update_mu(mu1_sub, cov1_sub, sample)
                mu2_con = func.update_mu(mu2_sub, cov2_sub, sample)
                samples = func.generate_biGP_samples(mymap.phi_bi, mu1_con, mu2_con, cov1_con, cov2_con, S, method=decom_method)
                v_hat += np.sum(np.where(np.dot(samples, x) - T_temp <= 0, 1, 0)) / S

    return v_hat / N

def calc_post_prob_DOT(J, U, mymap, N, S, delta, model='G', decom_method='cholesky'):
    ''' 
    evaluate the performance of a DOT calculated routing policy in terms of its posterior probability (the same way as how GP4 is evaluated).
    '''

    path_0 = U[mymap.r_0,0]
    max_time = U.shape[1] - 1

    node_1 = func.find_next_node(mymap, mymap.r_0, path_0)

    if node_1 == mymap.r_s:
        return J[mymap.r_0,0]

    rng = np.random.default_rng()
    v_hat = 0

    if model == 'G':
        mu_sub, cov_sub, cov_con = func.update_param(mymap.mu, mymap.cov, path_0)
        for i in range(N):
            sample_i = np.random.normal(mu_sub[2], np.sqrt(cov_sub[22]))
            time_i = np.ceil(sample_i/delta).astype(int)
            if max_time > time_i:
                mu_con = func.update_mu(mu_sub, cov_sub, sample_i)
                samples = rng.multivariate_normal(mu_con.reshape(-1), cov_con, S, method=decom_method)
                v_temp = 0
                for j in range(S):
                    curr_node = node_1
                    sample = sample_i
                    curr_time = time_i
                    while True:
                        next_link = U[curr_node, curr_time]
                        if next_link == -1:
                            break
                        curr_node = func.find_next_node(mymap, curr_node, next_link)
                        if next_link > path_0:
                            next_link -= 1
                        sample += samples[j, next_link]
                        curr_time = np.ceil(sample/delta).astype(int)
                        if max_time < curr_time:
                            break
                        elif max_time == curr_time and curr_node != mymap.r_s:
                            break
                        elif curr_node == mymap.r_s:
                            v_temp += 1
                            break
                v_hat += v_temp / S

    if model == 'log':
        mu_sub, cov_sub, cov_con = func.update_param(mymap.mu, mymap.cov, path_0)
        for i in range(N):
            sample_i = np.random.normal(mu_sub[2], np.sqrt(cov_sub[22]))
            time_i = np.ceil(np.exp(sample_i)/delta).astype(int)
            if max_time > time_i:
                mu_con = func.update_mu(mu_sub, cov_sub, sample_i)
                samples = np.exp(rng.multivariate_normal(mu_con.reshape(-1), cov_con, S, method=decom_method))
                v_temp = 0
                for j in range(S):
                    curr_node = node_1
                    sample = np.exp(sample_i)
                    curr_time = time_i
                    while True:
                        next_link = U[curr_node, curr_time]
                        if next_link == -1:
                            break
                        curr_node = func.find_next_node(mymap, curr_node, next_link)
                        if next_link > path_0:
                            next_link -= 1
                        sample += samples[j, next_link]
                        curr_time = np.ceil(sample/delta).astype(int)
                        if max_time < curr_time:
                            break
                        elif max_time == curr_time and curr_node != mymap.r_s:
                            break
                        elif curr_node == mymap.r_s:
                            v_temp += 1
                            break
                v_hat += v_temp / S

    if model == 'bi':
        mu1_sub, cov1_sub, cov1_con = func.update_param(mymap.mu, mymap.cov, path_0)
        mu2_sub, cov2_sub, cov2_con = func.update_param(mymap.mu2, mymap.cov2, path_0)
        for i in range(N):
            sample_i = func.generate_biGP_samples(mymap.phi_bi, mu1_sub[2], mu2_sub[2], cov1_sub[22], cov2_sub[22], 1).item()
            time_i = np.ceil(sample_i/delta).astype(int)
            if max_time > time_i:
                mu1_con = func.update_mu(mu1_sub, cov1_sub, sample_i)
                mu2_con = func.update_mu(mu2_sub, cov2_sub, sample_i)
                samples = func.generate_biGP_samples(mymap.phi_bi, mu1_con, mu2_con, cov1_con, cov2_con, S, method=decom_method)
                v_temp = 0
                for j in range(S):
                    curr_node = node_1
                    sample = sample_i
                    curr_time = time_i
                    while True:
                        next_link = U[curr_node, curr_time]
                        if next_link == -1:
                            break
                        curr_node = func.find_next_node(mymap, curr_node, next_link)
                        if next_link > path_0:
                            next_link -= 1
                        sample += samples[j, next_link]
                        curr_time = np.ceil(sample/delta).astype(int)
                        if max_time < curr_time:
                            break
                        elif max_time == curr_time and curr_node != mymap.r_s:
                            break
                        elif curr_node == mymap.r_s:
                            v_temp += 1
                            break
                v_hat += v_temp / S

    return v_hat / N