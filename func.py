import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ortho_group
from heapq import heapify, heappush, heappop
import networkx as nx
from itertools import islice

class Map():
    def __init__(self, model=None, decom=None):
        self.model = model
        self.decom = decom

    def make_map_with_M(self, mu, cov, M, mu2=None, cov2=None, phi_bi=None, is_multi=True):
        self.mu = mu
        self.cov = cov
        self.M = M
        self.n_node = self.M.shape[0]
        self.n_link = self.M.shape[1]
        self.mu2 = mu2
        self.cov2 = cov2
        self.phi_bi = phi_bi
        self.G = convert_map2graph(self, is_multi)

    def make_map_with_G(self, mu, cov, G, OD_true, mu2=None, cov2=None, phi_bi=None):
        self.mu = mu
        self.cov = cov
        self.r_0, self.r_s = OD_true[0], OD_true[1]
        self.mu2 = mu2
        self.cov2 = cov2
        self.phi_bi = phi_bi
        self.G = G
        self.M = None
        self.b = None

    def update_OD(self, OD_ori):
        self.b, self.r_0, self.r_s = generate_b(self.n_node, OD_ori[0], OD_ori[1])
        self.dij_cost, self.dij_path, self.dij_onehot_path = dijkstra(self.G, self.r_0, self.r_s)

    def generate_simple_map(self, model):
        if model == 'G':
            M = generate_simple_M(1)
            mu = np.array([10, 10.1, 10.2, 20]).reshape(-1, 1)
            cov = np.array([[2, -1, 1, 0], [-1, 2, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]])
            self.make_map_with_M(mu, cov, M)
            self.model = "G"
            self.decom = "cholesky"
            self.update_OD([1,3])

        elif model == 'log':
            M = generate_simple_M(1)
            mu = np.array([10, 10.1, 10.2, 20]).reshape(-1, 1)
            cov = np.array([[2, -1, 1, 0], [-1, 2, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]])
            mu_log, cov_log = calc_logGP4_param(mu, cov)
            self.make_map_with_M(mu_log, cov_log, M)
            self.model = "log"
            self.decom = "eigh"
            exp_mu = calc_exp_gauss(self.mu, self.cov)
            self.G = update_graph_weight(self.G, exp_mu)
            self.update_OD([1,3])

        elif model == 'bi':
            M = generate_simple_M(1)
            mu1 = np.array([14, 13.1, 8.2, 22]).reshape(-1, 1)
            cov1 = np.array([[2, -1, 1, 0], [-1, 2, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]])
            mu2 = np.array([6, 7.1, 12.2, 18]).reshape(-1, 1)
            cov2 = np.array([[1, 0.8, -0.7, 0], [0.8, 2, -0.5, 0], [-0.7, -0.5, 3, 0], [0, 0, 0, 2]])
            phi_bi = 0.5
            self.make_map_with_M(mu=mu1, cov=cov1, M=M, mu2=mu2, cov2=cov2, phi_bi=phi_bi)
            self.model = "bi"
            self.decom = "cholesky"
            mu_bi = calc_bi_gauss(self.phi_bi, self.mu, self.mu2)
            self.G = update_graph_weight(self.G, mu_bi)
            self.update_OD([1,3]) 

    def generate_real_map(self, map_id, map_dir):
        ''' map_id is an integer that identifies the map you wish to use.
            map_dir is the directory you store the networks, which can be download from the link provided in README.md.
            map_id | network
                0    Sioux Falls
                1    Anaheim
                2    Barcelona
                3    Chicago-Sketch
                4    Chengdu-Weekend Off-peak Hour
                5    Chengdu-Weekend Peak Hour
                6    Chengdu-Weekday Off-peak Hour
                7    Chengdu-Weekday Peak Hour
        '''
        M, mu, cov = extract_map(map_id, map_dir)
        self.make_map_with_M(mu, cov, M, is_multi=False)
        self.model = "G"
        self.decom = "eigh" if map_id == 0 else "cholesky"

class priority_dict(dict):
    def __init__(self, *args, **kwargs):
        super(priority_dict, self).__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in self.items()]
        heapify(self._heap)

    def smallest(self):
        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heappop(heap)
            v, k = heap[0]
        return k

    def get(self):
        heap = self._heap
        v, k = heappop(heap)
        while k not in self or self[k] != v:
            v, k = heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        super(priority_dict, self).__setitem__(key, val)

        if len(self._heap) < 2 * len(self):
            heappush(self._heap, (val, key))
        else:
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        super(priority_dict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def sorted_iter(self):
        while self:
            yield self.pop_smallest()

def generate_simple_M(n):                       # n: num of "loop" structure
    M = np.zeros((n+2,2*n+2))
    M[0,0] = 1
    M[1,0] = -1
    M[0,2*n+1] = 1
    M[n+1,2*n+1] = -1
    for i in range(0,n):
        M[i+1,2*i+1] = 1
        M[i+1,2*i+2] = 1
        M[i+2,2*i+1] = -1
        M[i+2,2*i+2] = -1

    return M

def generate_b(n_node, origin, destination):
    '''
    OD start from 1 when displayed or inputted, but start from 0 when stored and calculated.
    '''
    b = np.zeros(n_node)

    r_0 = origin-1
    r_s = destination-1

    b[r_0] = 1
    b[r_s] = -1

    return b.reshape(-1,1), r_0, r_s

def update_map(M, b, link, curr_node, next_node):
    M_temp = np.delete(M,link,axis=1)
    b_temp = np.copy(b)
    b_temp[curr_node] = 0
    b_temp[next_node] = 1
    return M_temp, b_temp

def update_param(mu, cov, link):
    mu_1 = np.delete(mu,link,axis=0)
    mu_2 = mu[link][0]
    mu_sub = {1:mu_1, 2:mu_2}

    cov_11 = np.delete(np.delete(cov,link,axis=1),link,axis=0)
    cov_12 = np.delete(cov[:,link],link,axis=0).reshape(-1,1)
    cov_21 = np.delete(cov[link,:],link).reshape(1,-1)
    cov_22 = cov[link,link]
    cov_sub = {11:cov_11, 12:cov_12, 21:cov_21, 22:cov_22}

    cov_con = cov_11-np.matmul(cov_12,cov_21)/cov_22

    return mu_sub, cov_sub, cov_con

def update_mu(mu_sub, cov_sub, sample):
    return mu_sub[1]+(sample-mu_sub[2])/cov_sub[22]*cov_sub[12]

def calc_exp_gauss(mu_log, cov_log):
    cov_diag = np.diag(cov_log).reshape(-1,1) if type(cov_log) is np.ndarray else cov_log
    exp_mu = np.exp(mu_log+cov_diag/2)
    return exp_mu

def calc_bi_gauss(phi, mu1, mu2):
    return phi*mu1+(1-phi)*mu2

def generate_cov1(mu, nu, factors):         #factors up, corr down
    '''
    Generate a semi-positive definite covariance matrix.
    'nu' controls the magnitude of variance and covariance.
    The correlation coefficients can be adjusted by changing 'factors'.
    When the 'factors' goes up, the magnitude of the corrlation coefficients generallty goes down
    '''
    n_link = np.size(mu)

    W = np.random.randn(n_link,factors)
    S = np.dot(W,W.T) + np.diag(np.random.rand(1,n_link))
    corr = np.matmul(np.matmul(np.diag(1/np.sqrt(np.diag(S))),S),np.diag(1/np.sqrt(np.diag(S))))

    sigma = nu*mu#*np.random.rand(n_link,1).reshape(-1,1)
    sigma = np.matmul(sigma,sigma.T)
    cov = sigma*corr

    return corr, sigma, cov

def generate_cov2(mu, nu):
    '''
    Generate a positive definite covariance matrix.
    'nu' controls the magnitude of variance and covariance.
    '''
    n_link = np.size(mu)

    D = np.diag(np.random.rand(n_link))
    U = ortho_group.rvs(dim=n_link)
    S = np.matmul(np.matmul(U.T,D),U)
    corr = np.matmul(np.matmul(np.diag(1/np.sqrt(np.diag(S))),S),np.diag(1/np.sqrt(np.diag(S))))
    
    sigma = nu*mu#*np.random.rand(n_link,1).reshape(-1,1)
    sigma = np.matmul(sigma,sigma.T)
    cov = sigma*corr

    return corr, sigma, cov

def calc_logGP4_param(mu_ori, cov_ori):
    cov_log = np.log(cov_ori/np.dot(mu_ori,mu_ori.T)+1)
    mu_log = np.log(mu_ori)-0.5*np.diag(cov_log).reshape(-1,1)

    return mu_log, cov_log

def generate_biGP_samples(phi_bi, mu1, mu2, cov1, cov2, S, method='cholesky'):
    rng = np.random.default_rng()
    if np.size(mu1) > 1:
        samples1 = rng.multivariate_normal(mu1.reshape(-1), cov1, S, method=method)
        samples2 = rng.multivariate_normal(mu2.reshape(-1), cov2, S, method=method)
        dim = mu1.size
    else:
        samples1 = np.random.normal(mu1, np.sqrt(cov1), [S,1])
        samples2 = np.random.normal(mu2, np.sqrt(cov2), [S,1])
        dim = 1

    phi1 = np.where(np.random.rand(S, dim) < phi_bi, 1, 0)
    phi2 = np.ones(phi1.shape)-phi1
    samples = np.multiply(phi1,samples1) + np.multiply(phi2,samples2)

    return samples

def generate_samples(mymap, S):
    '''
    return: N*S matrix
    '''
    rng = np.random.default_rng()
    if mymap.model == "G":
        samples = rng.multivariate_normal(mymap.mu.reshape(-1), mymap.cov, S, method=mymap.decom)
    elif mymap.model == "log":
        samples = rng.multivariate_normal(mymap.mu.reshape(-1), mymap.cov, S, method=mymap.decom)
        samples = np.exp(samples)
    elif mymap.model == "bi":
        samples = generate_biGP_samples(mymap.phi_bi, mymap.mu, mymap.mu2, mymap.cov, mymap.cov2, S, method=mymap.decom)
    return samples.T
    
def sort_path_order(path, mymap):
    if type(path) is np.ndarray:
        path = path.tolist()
    sorted_path = []
    node = mymap.r_0
    while node != mymap.r_s:
        for link in path:
            if mymap.M[node,link] == 1:
                sorted_path.append(link)
                node = np.where(mymap.M[:,link]==-1)[0].item()
                path.remove(link)
                break
    return np.array(sorted_path)

def first_path_link(path, mymap):
    if type(path) is np.ndarray:
        path = path.tolist()
    node = mymap.r_0
    for link in path:
        if mymap.M[node,link] == 1:
            sorted_path=[link]
            path.remove(link)
            break
    return np.array(sorted_path+path)

def convert_node2onehot(path, G):
    link_ids = []
    node_pairs=zip(path[0:],path[1:])

    for u,v in node_pairs:
        edge = sorted(G[u][v], key=lambda x:G[u][v][x]['weight'])
        link_ids.append(G[u][v][edge[0]]['index'])

    onehot = np.zeros(G.size())
    onehot[link_ids] = 1
    onehot = onehot.reshape(-1,1)

    return link_ids, onehot

def convert_map2graph(mymap, is_multi=True):
    G = nx.MultiDiGraph() if is_multi else nx.DiGraph()

    for i in range(mymap.M.shape[1]):
        start = np.where(mymap.M[:,i]==1)[0].item()
        end = np.where(mymap.M[:,i]==-1)[0].item()
        G.add_edge(start, end, weight=mymap.mu[i].item(), index=i)

    return G

def update_graph_weight(G, new_mu):
    is_multi = True if G.__module__ == 'networkx.classes.multidigraph' else False

    G_new = G.copy()
    if is_multi:
        for u,v,k,d in G.edges(data=True, keys=True):
            G_new[u][v][k]['weight'] = new_mu[d['index']].item()
    else:
        for u,v,d in G.edges(data=True):
            G_new[u][v]['weight'] = new_mu[d['index']].item()
    return G_new

def remove_graph_edge(G, e_id):
    is_multi = True if G.__module__ == 'networkx.classes.multidigraph' else False

    G_new = G.copy()
    if is_multi:
        for u,v,k,d in G.edges(data=True, keys=True):
            if d['index'] == e_id:
                G_new.remove_edge(u,v,k)
            elif d['index'] > e_id:
                G_new[u][v][k]['index'] -= 1
    else:
        for u,v,d in G.edges(data=True):
            if d['index'] == e_id:
                G_new.remove_edge(u,v)
            elif d['index'] > e_id:
                G_new[u][v]['index'] -= 1
    return G_new

def find_next_node(mymap, curr_node, link_idx):
    for _, next_node, d in mymap.G.out_edges(curr_node, data=True):
        if d['index'] == link_idx:
            return next_node

def dijkstra(G, start, end, ext_weight=None):
    if not G.has_node(start) or not G.has_node(end):
        return -1, None, None

    cost = {}
    for node in G.nodes():
        cost[node] = float('inf')
    cost[start] = 0
    prev_node = {start: None}
    prev_edge = {start: None}
    PQ = priority_dict(cost)

    while bool(PQ):
        curr_node = PQ.get()

        if curr_node == end:
            break
        
        for _, next_node, d in G.out_edges(curr_node, data=True):
            if next_node in PQ:
                alt = cost[curr_node] + (d['weight'] if ext_weight is None else ext_weight[d['index']].item())
                if alt < cost[next_node]:
                    cost[next_node] = alt
                    prev_node[next_node] = curr_node
                    prev_edge[next_node] = d['index']
                    PQ[next_node] = alt

    if curr_node == end and end in prev_node:
        path_cost = cost[end]
        path = []
        while curr_node != start:
            path.append(prev_edge[curr_node])
            curr_node = prev_node[curr_node]
        path.reverse()

        onehot = np.zeros(G.size())
        onehot[path] = 1
        onehot = onehot.reshape(-1, 1)
        return path_cost, path, onehot
    else:
        return -1, None, None

def k_shortest_paths(mymap, k, weight="weight"):
    return list(islice(nx.shortest_simple_paths(mymap.G, mymap.r_0, mymap.r_s, weight=weight), k))

def t_test(x, y, alternative='greater', alpha=0.05):
    t_stat, double_p = stats.ttest_ind(x,y,equal_var = False)

    if alternative == 'both-sided':
        pval = double_p
    elif alternative == 'greater':
        if t_stat > 0:
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    elif alternative == 'less':
        if t_stat < 0:
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.

    return pval, pval<alpha

def generate_OD_pairs(mymap, n_pair):
    def generate_OD(n_node):
        r_0 = np.random.randint(n_node) + 1
        while not mymap.G.has_node(r_0-1):
            r_0 = np.random.randint(n_node) + 1
        r_s = np.random.randint(n_node) + 1
        while r_s == r_0 or not mymap.G.has_node(r_s-1):
            r_s = np.random.randint(n_node) + 1
        OD = [r_0, r_s]
        return OD

    OD_pairs = []
    count = 0

    while count < n_pair:
        OD = generate_OD(mymap.n_node)
        while OD in OD_pairs or dijkstra(mymap.G, OD[0]-1, OD[1]-1)[0] == -1:
            OD = generate_OD(mymap.n_node)
        OD_pairs.append(OD)
        count += 1

    return OD_pairs
	
def extract_map(map_id, map_dir):
    map_list = ['SiouxFalls', 'Anaheim', 'Barcelona', 'Chicago_Sketch', 'Weekend_Offpeak', 'Weekend_Peak', 'Weekday_Offpeak', 'Weekday_Peak']

    if map_id < 4:
        map_dir += map_list[map_id] + '/'
    else:
        map_dir += 'Chengdu/'

    raw_map_data = pd.read_csv(map_dir + map_list[map_id] + '_network.csv')

    origins = raw_map_data['From']
    destinations = raw_map_data['To']
    if origins.min() == 0 or destinations.min() == 0:
        origins += 1
        destinations += 1
    n_node = max(origins.max(), destinations.max())
    n_link = raw_map_data.shape[0]

    M = np.zeros((n_node,n_link))
    for i in range(n_link):
        M[origins[i]-1,i] = 1
        M[destinations[i]-1,i] = -1

    mu = np.array(raw_map_data['Cost']).reshape(-1,1)
    
    cov = np.load(map_dir + map_list[map_id] + '_cov.npy')

    # if map_id == 0:
    #     cov = generate_cov2(mu, 0.5)[2]

    return M, mu, cov

def record(alg, res, file_name):
    print(alg + " mean, std, t, t_max: " + str(res) + '\n')

    fp = open(file_name, 'a+')
    fp.write(alg + " mean, std, t, t_max: " + str(res) + "\n")
    fp.close()

def write_file(content_name, content, file_name):
    fp = open(file_name, 'a+')
    fp.write("{}={}\n".format(content_name,content))
    fp.close()
