import numpy as np
import pandas as pd
import func
import time
from evaluation import calc_path_prob, calc_post_prob
from scipy.stats import norm
import gurobipy as gp
from gurobipy import GRB

def PLM(mymap, S, T, phi=10, e=0.1):
    g_best = -10**7
    g_best_last = -10**7
    probability_last = 0
    max_path = 0
    lmd = np.random.random([S, 1])
    
    samples = func.generate_samples(mymap, S)
    
    T = np.ones([S, 1]) * T

    k = 1
    k_x = 0

    while(True):
        d_cost, path, x = func.dijkstra(mymap.G, mymap.r_0, mymap.r_s, ext_weight=np.dot(samples, lmd))
        sub1_cost = d_cost - np.dot(T.T, lmd)

        tmp = np.ones([S, 1])-lmd
        xi = np.where(tmp > 0, 0, 10**7)
        sub2_cost = np.sum(np.dot(tmp.T, xi))

        cost = sub1_cost + sub2_cost
        probability = np.sum(np.where(np.dot(samples.T, x) <= T, 1, 0)) / samples.shape[1]

        if probability >= probability_last:
            max_path = path
            probability_last = probability
        probability = max(probability, probability_last)
        # print(k)
        # print(cost)
        # print(max_path)

        g_best = max(cost, g_best)
        if (g_best - g_best_last >= e):
            k_x = k
        g_best_last = g_best

        if(k-k_x >= phi):
            break

        d_g = np.dot(samples.T, x) - T - xi

        alpha = 0.0001/np.sqrt(k)
        lmd += alpha * d_g
        lmd = np.where(lmd > 0, lmd, 0)
        
        k += 1

    # print("final path:" + str(np.array(max_path) + 1))
    return probability, max_path

def ILP(mymap, S, T):
    V = 10**4

    samples = func.generate_samples(mymap, S).T

    obj_temp1 = np.zeros(mymap.n_link)
    obj_temp2 = np.ones(S)
    obj = np.hstack((obj_temp1, obj_temp2))

    eq_temp = np.zeros([mymap.n_node, S])
    eq_constr = np.hstack((mymap.M, eq_temp))

    ineq_temp = -V * np.eye(S)
    ineq_constr = np.hstack((samples, ineq_temp))

    T = np.ones(S) * T

    n_elem = mymap.n_link + S

    m = gp.Model("ilp")
    m.Params.LogToConsole = 0

    z = m.addMVar(shape=n_elem, vtype=GRB.BINARY, name="z")
    m.setObjective(obj @ z, GRB.MINIMIZE)
    m.addConstr(ineq_constr @ z <= T, name="ineq")
    m.addConstr(eq_constr @ z == mymap.b.reshape(-1), name="eq")
    m.optimize()

    res = z.X

    prob = 1 - np.dot(obj.T, res).item()/S
    path = np.flatnonzero(res[:mymap.n_link])
    path = func.sort_path_order(path, mymap)
    # print(prob)

    # print("final path:" + str(path + 1))
    return prob, path

def MIP_LR(mymap, S, T, phi=5, e=1):
    g_best = -10**7
    g_best_last = -10**7
    up_last = 0
    up_path = 0
    M = 10**4
    rho = np.random.random()
    lmd = np.random.random([S, 1])

    samples = func.generate_samples(mymap, S)

    T = np.ones([S, 1]) * T

    k = 1
    k_x = 0

    while(True):
        sigma = 1 if 1-rho >= 0 else 0
        sub1_cost = min(0, 1-rho)

        tmp = -M*lmd + rho/S
        z_w = np.where(tmp >= 0, 0, 1)
        sub2_cost = np.dot(tmp.T, z_w)

        paths = []
        d_cost_total = 0
        
        phys_cost = np.zeros([S, 1])
        path_prob = np.zeros([S, 1])
        for w in range(S):
            samples_tmp = lmd[w,0]*samples[:,w].reshape(-1,1)
            d_cost, path, x = func.dijkstra(mymap.G, mymap.r_0, mymap.r_s, ext_weight=samples_tmp)
            phys_cost[w,0] = np.dot(samples[:,w],x)
            path_prob[w,0] = np.sum(np.where(np.dot(samples.T, x) <= T, 1, 0))/S
            paths.append(path)
            d_cost_total += d_cost

        sub3_cost = d_cost_total - np.dot(T.T, lmd)

        cost = sub1_cost + sub2_cost + sub3_cost
        if np.max(path_prob) >= up_last:
            up = np.max(path_prob)
            up_path = paths[np.argmax(path_prob)]
            up_last = up
        # print(k)
        # print(cost)
        # print(up_path)
        
        probability = 1 - np.sum(z_w)/S

        g_best = max(cost, g_best)
        if (g_best - g_best_last >= e):
            k_x = k
        g_best_last = g_best

        if(k-k_x >= phi):
            break
        
        d_rho = sigma - probability
        d_lmd = -M*z_w - T + phys_cost

        alpha = 0.00001 / np.sqrt(k)
        rho += alpha * d_rho
        rho = max(0, rho)
        lmd += alpha * d_lmd
        lmd = np.where(lmd > 0, lmd, 0)

        k += 1

    # print("final path:" + str(np.array(up_path) + 1))
    return up, up_path

def other_iterations(alg, mymap, T, N, S, MaxIter, is_posterior=False):
    '''
    Run a certain algorithm, 'alg', for 'MaxIter' times and return the statistics.
    When 'is_posterior' is True, the performance of the path is evaluated in terms of its posterior probability (the same as how GP4 is evaluated).
    '''
    pro = []
    t_delta = []

    for ite in range(MaxIter):
        print('{} iteration #{}'.format(alg.__name__, ite))
        t1 = time.perf_counter()
        prob, path = alg(mymap, S, T)
        t_delta.append(time.perf_counter() - t1)
        print("final path: {}\n".format(str(np.array(path) + 1)))
        if is_posterior: 
            if len(path) == 1:
                pro.append(prob)
            else:
                pro.append(calc_post_prob(path, mymap, T, N, S)) 
        else:
            pro.append(prob)
    
    return np.mean(pro), np.std(pro, ddof=1), np.mean(t_delta), np.max(t_delta)

class DOT:
    def __init__(self, mymap, T, delta, samples=None):
        self.map = mymap
        self.T = T
        self.delta = delta #delta = 0.05 for Chicago-sketch, delta = 2 for Chengdu
        self.model = mymap.model

        self.samples = samples #you might want to use a set of unchanged samples to evaluate the performance

        t1 = time.perf_counter()
        self.DOT_Policy()
        self.DOT_t_delta = time.perf_counter() - t1

    def DOT_Policy(self):
        '''
        Generate DOT routing policy.
        '''
        n_timestamp = np.ceil(self.T/self.delta).astype(int)
        self.delta = self.T / n_timestamp
        J = np.zeros([self.map.n_node, n_timestamp+1])
        J[self.map.r_s,:] = 1
        U = -1 * np.ones([self.map.n_node, n_timestamp+1])
        times = np.linspace(0, self.T, num=n_timestamp+1).reshape(-1,1)

        if self.model == 'G':
            CDF = norm.cdf(times, self.map.mu.reshape(-1), np.sqrt(np.diag(self.map.cov)))
            # CDF[0] = 0 #uncomment this line for Chengdu
            self.CDF_delta = CDF[1:, :] - CDF[:n_timestamp, :]
        elif self.model == 'log':
            CDF = norm.cdf(np.log(times), self.map.mu.reshape(-1), np.sqrt(np.diag(self.map.cov)))
            # CDF[0] = 0 #uncomment this line for Chengdu
            self.CDF_delta = CDF[1:, :] - CDF[:n_timestamp, :]
        elif self.model == 'bi':
            CDF1 = norm.cdf(times, self.map.mu.reshape(-1), np.sqrt(np.diag(self.map.cov)))
            # CDF1[0] = 0 #uncomment this line for Chengdu
            CDF1_delta = CDF1[1:, :] - CDF1[:n_timestamp, :]
            CDF2 = norm.cdf(times, self.map.mu2.reshape(-1), np.sqrt(np.diag(self.map.cov2)))
            # CDF2[0] = 0 #uncomment this line for Chengdu
            CDF2_delta = CDF2[1:, :] - CDF2[:n_timestamp, :]
            self.CDF_delta = func.calc_bi_gauss(self.map.phi_bi, CDF1_delta, CDF2_delta)


        for timestamp in range(n_timestamp-1,-1,-1):
            for node in self.map.G.nodes:
                if node != self.map.r_s:
                    prob_max = 0
                    u = -1
                    for _, next_node, d in self.map.G.out_edges(node, data=True):
                        link_idx = d['index']
                        prob = np.dot(self.CDF_delta[:n_timestamp-timestamp, link_idx], J[next_node, timestamp+1:n_timestamp+1])
                        if prob >= prob_max:
                            prob_max = prob
                            u = link_idx
                    
                    J[node, timestamp] = prob_max
                    U[node, timestamp] = u

        self.J = J
        self.U = U.astype(int)

    def get_DOT_prob(self, t=None):
        '''
        Return the DOT probability given certain deadline 't'.
        Note that t < T.
        '''
        if t is None:
            t = self.T
        stamp = self._t2stamp(t)
        return self.J[self.map.r_0, stamp].item()

    def get_DOT_post(self, t, N, S):
        '''
        Return the DOT posterior performance given certain deadline 't'.
        Note that t < T.
        '''
        stamp = self._t2stamp(t)
        J = self.J[:,stamp:]
        U = self.U[:,stamp:]
        return self._calc_DOT_post(J, U, N, S)

    def PA(self, t, N=64, S=128, is_posterior=False, maxspeed=3):
        '''
        Return the performance of Pruning Algorithm. 
        If 'is_posterior' is True, the posterior performance of pruning algorithm calculated path is returned.
        'N', 'S' are only valid when 'is_posterior' is True.
        'maxspeed' is used during the calculation of lower bound 1, 
        where the minimum travel time of a link is defined as mu/maxspeed.
        '''
        stamp = self._t2stamp(t)
        J = self.J[:,stamp:]
        self.maxspeed = maxspeed

        t1 = time.perf_counter()
        self.PA_prob, path = self._PA(t, J)
        self.PA_t_delta = time.perf_counter() - t1

        if is_posterior:
            if len(path) == 1:
                return self.PA_prob
            else:
                return calc_post_prob(path, self.map, t, N, S)
        else:
            return self.PA_prob

    def _PA(self, t, J):
        lb = calc_path_prob(self.map.dij_path, self.map, t, self.samples)
        #lb *= 0.5 #you might want to relax the lower bound since the original bound might be too strict to generate a feasible path

        empty_df = pd.DataFrame(columns=['pre_node', 'pre_sub', 'link_idx', 'min_time'])
        df = pd.DataFrame({"pre_node":None, "pre_sub":None, "link_idx":None, "min_time":0}, index=[0])
        self.PI = {self.map.r_0:df}
        L = [self.map.r_0]

        while L:
            pre_node = L.pop(0)

            for _, next_node, d in self.map.G.out_edges(pre_node, data=True):
                is_empty = 0
                if next_node not in self.PI:
                    self.PI[next_node] = empty_df
                if self.PI[next_node].empty:
                    is_empty = 1
                link_idx = d['index']
                modified_flag = 0

                for pre_sub, row in self.PI[pre_node].iterrows():
                    if not is_empty:
                        if self._is_cyclic(next_node, pre_node, pre_sub):
                            continue
                        if self._is_explored(next_node, pre_node, pre_sub, link_idx):
                            continue

                    df = self._add_subpath(pre_node, pre_sub, link_idx, row['min_time'])
                    if df["min_time"] >= J.shape[1]:
                        continue

                    ub1 = J[next_node, df["min_time"]]
                    if ub1 < lb:
                        continue
                    ub2 = self._calc_ub2(next_node, pre_node, pre_sub, link_idx, J)
                    #ub2 *= 1.2 #you might want to relax the upper bound 2 since the original bound might be too strict to generate a feasible path
                    if lb > ub2:
                        continue

                    self.PI[next_node] = self.PI[next_node].append(df, ignore_index=True)
                    modified_flag = 1

                if modified_flag and next_node not in L:
                    L.append(next_node)

        #Extract candidate path
        paths = []
        probs = []

        if self.map.r_s not in self.PI or self.PI[self.map.r_s].empty:
            print("Warning!!!! No feasible path is found by Pruning algorithm. \
            The path with maximum on-time-arrival probabolity among K shortest path is returned")
            if self.map.G.__module__ == 'networkx.classes.multidigraph':
                paths.append(self.map.dij_path)
                probs.append(calc_path_prob(self.map.dij_path, self.map, t, self.samples))
            else:
                for path_node in func.k_shortest_paths(self.map, k=5):
                    path = []
                    for j in range(len(path_node)-1):
                        path.append(self.map.G[path_node[j]][path_node[j+1]]['index'])

                    prob = calc_path_prob(path, self.map, t, self.samples)
                    paths.append(path)
                    probs.append(prob)
                    # print(path)
                    # print(prob)
        else:
            for _, temp in self.PI[self.map.r_s].iterrows():
                pre_node = temp["pre_node"]
                pre_sub = temp["pre_sub"]
                path = []

                while pre_node is not None:
                    path.append(temp['link_idx'])
                    temp = self.PI[pre_node].loc[pre_sub]
                    pre_node = temp['pre_node']
                    pre_sub = temp['pre_sub']
                
                path.reverse()
                prob = calc_path_prob(path, self.map, t, self.samples)
                paths.append(path)
                probs.append(prob)
                # print(path)
                # print(prob)

        #Find the Path with MPOA
        MPOA = np.max(probs)
        MPOA_path = paths[np.argmax(probs)]
        
        print("Pruning\nfinal path: {}\n".format(str(np.array(MPOA_path) + 1)))

        return MPOA, MPOA_path

    def _is_cyclic(self, next_node, pre_node, pre_sub):
        while pre_node is not None:
            if next_node == pre_node:
                return True
            temp = self.PI[pre_node].loc[pre_sub]
            pre_node = temp['pre_node']
            pre_sub = temp['pre_sub']
        return False
        
    def _is_explored(self, next_node, pre_node, pre_sub, link_idx):
        df = self.PI[next_node]
        return not df[(df["pre_node"] == pre_node) & (df["pre_sub"] == pre_sub) & (df["link_idx"] == link_idx)].empty

    def _calc_ub2(self, next_node, pre_node, pre_sub, link_idx, J):
        H = J[next_node, :].reshape(-1)
        # size = H.size-1
        # mask = np.flip(np.triu(np.ones((size, size)), k=0), axis=1)
        while pre_node is not None:
            H_new = np.zeros_like(H)
            for i in range(H.size-1):
                H_new[i] = np.dot(self.CDF_delta[:H.size-1-i, link_idx], H[i+1:])
                if H_new[i] < 1e-4:
                    break
            H = H_new

            temp = self.PI[pre_node].loc[pre_sub]
            pre_node = temp['pre_node']
            pre_sub = temp['pre_sub']
            link_idx = temp["link_idx"]

        return H[0]

    def _add_subpath(self, pre_node, pre_sub, link_idx, pre_min_time):
        min_time = pre_min_time + np.floor(self.map.mu[link_idx].item()/self.maxspeed/self.delta).astype(int)
        return pd.Series({"pre_node":pre_node, "pre_sub":pre_sub, "link_idx":link_idx, "min_time":min_time})

    def _t2stamp(self, t):
        return round((self.T - t)/self.delta)

    def _calc_DOT_post(self, J, U, N, S):
        ''' 
        Evaluate the performance of a DOT generated routing policy in terms of its posterior probability (the same way as how GP4 is evaluated).
        '''
        path_0 = U[self.map.r_0,0]
        max_time = U.shape[1] - 1

        node_1 = func.find_next_node(self.map, self.map.r_0, path_0)

        if node_1 == self.map.r_s:
            return J[self.map.r_0,0]

        rng = np.random.default_rng()
        v_hat = 0

        if self.model == 'G':
            mu_sub, cov_sub, cov_con = func.update_param(self.map.mu, self.map.cov, path_0)
            for i in range(N):
                sample_i = np.random.normal(mu_sub[2], np.sqrt(cov_sub[22]))
                time_i = np.ceil(sample_i/self.delta).astype(int)
                if max_time > time_i:
                    mu_con = func.update_mu(mu_sub, cov_sub, sample_i)
                    samples = rng.multivariate_normal(mu_con.reshape(-1), cov_con, S, method=self.map.decom)
                    v_temp = 0
                    for j in range(S):
                        curr_node = node_1
                        sample = sample_i
                        curr_time = time_i
                        cnt = 0
                        while True:
                            next_link = U[curr_node, curr_time]
                            if next_link == -1:
                                break
                            curr_node = func.find_next_node(self.map, curr_node, next_link)
                            if next_link > path_0:
                                next_link -= 1
                            sample += samples[j, next_link]
                            # sample = max(0,sample) #uncomment this line for Chengdu
                            curr_time = np.ceil(sample/self.delta).astype(int)
                            cnt += 1
                            if cnt == 100:
                                break
                            if max_time < curr_time:
                                break
                            elif max_time == curr_time and curr_node != self.map.r_s:
                                break
                            elif curr_node == self.map.r_s:
                                v_temp += 1
                                break
                    v_hat += v_temp / S

        elif self.model == 'log':
            mu_sub, cov_sub, cov_con = func.update_param(self.map.mu, self.map.cov, path_0)
            for i in range(N):
                sample_i = np.random.normal(mu_sub[2], np.sqrt(cov_sub[22]))
                time_i = np.ceil(np.exp(sample_i)/self.delta).astype(int)
                if max_time > time_i:
                    mu_con = func.update_mu(mu_sub, cov_sub, sample_i)
                    samples = np.exp(rng.multivariate_normal(mu_con.reshape(-1), cov_con, S, method=self.map.decom))
                    v_temp = 0
                    for j in range(S):
                        curr_node = node_1
                        sample = np.exp(sample_i)
                        curr_time = time_i
                        while True:
                            next_link = U[curr_node, curr_time]
                            if next_link == -1:
                                break
                            curr_node = func.find_next_node(self.map, curr_node, next_link)
                            if next_link > path_0:
                                next_link -= 1
                            sample += samples[j, next_link]
                            curr_time = np.ceil(sample/self.delta).astype(int)
                            if max_time < curr_time:
                                break
                            elif max_time == curr_time and curr_node != self.map.r_s:
                                break
                            elif curr_node == self.map.r_s:
                                v_temp += 1
                                break
                    v_hat += v_temp / S

        elif self.model == 'bi':
            mu1_sub, cov1_sub, cov1_con = func.update_param(self.map.mu, self.map.cov, path_0)
            mu2_sub, cov2_sub, cov2_con = func.update_param(self.map.mu2, self.map.cov2, path_0)
            for i in range(N):
                sample_i = func.generate_biGP_samples(self.map.phi_bi, mu1_sub[2], mu2_sub[2], cov1_sub[22], cov2_sub[22], 1).item()
                time_i = np.ceil(sample_i/self.delta).astype(int)
                if max_time > time_i:
                    mu1_con = func.update_mu(mu1_sub, cov1_sub, sample_i)
                    mu2_con = func.update_mu(mu2_sub, cov2_sub, sample_i)
                    samples = func.generate_biGP_samples(self.map.phi_bi, mu1_con, mu2_con, cov1_con, cov2_con, S, method=self.map.decom)
                    v_temp = 0
                    for j in range(S):
                        curr_node = node_1
                        sample = sample_i
                        curr_time = time_i
                        cnt = 0
                        while True:
                            next_link = U[curr_node, curr_time]
                            if next_link == -1:
                                break
                            curr_node = func.find_next_node(self.map, curr_node, next_link)
                            if next_link > path_0:
                                next_link -= 1
                            sample += samples[j, next_link]
                            # sample = max(0,sample) #uncomment this line for Chengdu
                            curr_time = np.ceil(sample/self.delta).astype(int)
                            cnt += 1
                            if cnt == 100:
                                break
                            if max_time < curr_time:
                                break
                            elif max_time == curr_time and curr_node != self.map.r_s:
                                break
                            elif curr_node == self.map.r_s:
                                v_temp += 1
                                break
                    v_hat += v_temp / S

        return v_hat / N