import numpy as np
import pandas as pd
import cvxopt
from cvxopt import glpk
import os
from scipy.stats import ortho_group

def generate_A(n):                              # n: # of "loop" structure
    A = np.zeros((n+2,2*n+2))                   # n+2: # of nodes; 2n+2: # of links
    A[0,0] = 1
    A[1,0] = -1
    A[0,2*n+1] = 1
    A[n+1,2*n+1] = -1
    for i in range(0,n):
        A[i+1,2*i+1] = 1
        A[i+1,2*i+2] = 1
        A[i+2,2*i+1] = -1
        A[i+2,2*i+2] = -1

    A_idx = np.array(range(1,2*n+3))          # true index of links
    return A, A_idx

def generate_b(n_node, origin, destination):    # o and d count from 1, while store from 0
    b = np.zeros(n_node)

    r_0 = origin-1
    r_s = destination-1

    b[r_0] = 1
    b[r_s] = -1

    return b.reshape(-1,1), r_0, r_s

def generate_mu(n_link, mu_scaler=10):
    mu = mu_scaler*np.ones(n_link)
    # mu[0][np.random.randint(1,n_link-1)] += 0.1
    mu[-1] = (n_link/2)*mu_scaler

    # mu = np.random.rand(1,n_link)
    # mu[-1] = n_link/4.5
    return mu.reshape(-1,1)

def generate_sigma(n_link, sigma_scaler=1):
    D = sigma_scaler*np.diag(np.random.rand(n_link))
    U = ortho_group.rvs(dim=n_link)
    sigma = np.matmul(np.matmul(U.T,D),U)
    return sigma

def generate_map(n, origin=1, destination=None):
    if destination == None:
        destination = n+2
    n_node = n+2
    n_link = 2*n+2
    A, A_idx = generate_A(n)
    b, r_0, r_s = generate_b(n_node,origin,destination)
    return A, A_idx, b, r_0, r_s, n_link


# A, A_idx, b, r_0, r_s = generate_map(2)
# mu = generate_mu(6)
# sigma = generate_sigma(6)
# print(A)
# print(A_idx)
# print(b)
# print(mu)
# print(sigma)
# print(r_0)
# print(r_s)

def cvxopt_glpk_minmax(c, A, b, x_min=0, x_max=1):
    dim = np.size(c,0)

    x_min = x_min * np.ones(dim)
    x_max = x_max * np.ones(dim)
    G = np.vstack([+np.eye(dim),-np.eye(dim)])
    h = np.hstack([x_max, -x_min])
    # G = -np.eye(dim)
    # h = x_min.T

    c = cvxopt.matrix(c,tc='d')
    A = cvxopt.matrix(A,tc='d')
    b = cvxopt.matrix(b,tc='d')
    G = cvxopt.matrix(G,tc='d')
    h = cvxopt.matrix(h,tc='d')
    # sol = cvxopt.solvers.lp(c, G, h, A, b, solver='glpk',options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
    _,x = glpk.ilp(c,G,h,A,b,options={'msg_lev':'GLP_MSG_OFF'})

    return np.array(x)

def update_map(A, b, link, curr_node, next_node):
    A_temp = np.delete(A,link,axis=1)
    b_temp = np.copy(b)
    b_temp[curr_node] = 0
    b_temp[next_node] = 1
    return A_temp, b_temp

def update_param(mu, sigma, link):
    mu_1 = np.delete(mu,link,axis=0)
    mu_2 = mu[link][0]
    mu_sub = {1:mu_1, 2:mu_2}

    sigma_11 = np.delete(np.delete(sigma,link,axis=1),link,axis=0)
    sigma_12 = np.delete(sigma[:,link],link,axis=0).reshape(-1,1)
    sigma_21 = np.delete(sigma[link,:],link).reshape(1,-1)
    sigma_22 = sigma[link,link]
    sigma_sub = {11:sigma_11, 12:sigma_12, 21:sigma_21, 22:sigma_22}

    sigma_con = sigma_11-np.matmul(sigma_12,sigma_21)/sigma_22

    return mu_sub, sigma_sub, sigma_con

def update_mu(mu_sub, sigma_sub, sample):
    return mu_sub[1]+(sample-mu_sub[2])/sigma_sub[22]*sigma_sub[12]

def calc_exp_gauss(mu, sigma):
    sigma_diag = np.diag(sigma).reshape(-1,1) if type(sigma) is np.ndarray else sigma
    exp_mu = np.exp(mu+sigma_diag/2)
    return exp_mu

def calc_bi_gauss(phi, mu1, mu2):
    return phi*mu1+(1-phi)*mu2

# print(mu[1])
# print(type(sigma))
# ex = calc_exp_mu(mu, sigma)
# print(ex)
# A_temp, b_temp = update_map(A,b,3,0,3)
# mu_sub, sigma_sub, sigma_con = update_param(mu,sigma,3)

# print(A_temp)
# print(b_temp)
# print(mu_sub)
# print(sigma_sub)
# print(sigma_con)

# x = cvxopt_glpk_minmax(mu, A, b)
# print(x)


def extract_map(map_id):
    os.chdir('/Users/steve/Documents/CMC/TransportationNetworks-master/TransportationNetworks-master/') #change it to your own directory
    table_paths = ['SiouxFalls/SiouxFalls_network.xlsx', 
                'Anaheim/Anaheim_network.xlsx', 
                'Barcelona/Barcelona_network.xlsx', 
                'Chicago-Sketch/Chicago_Sketch_network.xlsx']

    raw_map_data = pd.read_excel(table_paths[map_id])

    origins = raw_map_data['From']
    destinations = raw_map_data['To']
    n_node = max(origins.max(), destinations.max())
    n_link = raw_map_data.shape[0]
    
    A = np.zeros((n_node,n_link))
    for i in range(n_link):
        A[origins[i]-1,i] = 1
        A[destinations[i]-1,i] = -1
        
    A_idx = np.arange(1,n_link+1)
    
    mu = np.array(raw_map_data['Cost']).reshape(-1,1)
        
    return A, A_idx, mu

def generate_cov(mu, nu):
    n_link = np.size(mu)
    
    sigma = nu*mu*np.random.rand(n_link,1)
    
    n_sample = n_link
    samples = np.zeros((n_link,n_sample))
    
    for i in range(np.shape(samples)[0]):
        for j in range (np.shape(samples)[1]):
            # while samples[i][j] <= 0:
            samples[i][j] = np.random.normal(mu[i],sigma[i])
    
    cov = np.cov(samples)
    
    return cov

def generate_cov1(mu, nu, factors):         #factors up, corr down
    n_link = np.size(mu)
        
    W = np.random.randn(n_link,factors)
    S = np.dot(W,W.T) + np.diag(np.random.rand(1,n_link))
    corr = np.matmul(np.matmul(np.diag(1/np.sqrt(np.diag(S))),S),np.diag(1/np.sqrt(np.diag(S))))
    
    sigma = nu*mu*np.random.rand(n_link,1).reshape(-1,1)
    
    sigma = np.matmul(sigma,sigma.T)

    cov = sigma*corr
    
    return corr, sigma, cov

def get_let_path(mu,A,b):
    sol = cvxopt_glpk_minmax(mu,A,b)
    if sol.all() == None:
        return None, None

    else:
        selected_links = list(np.where(sol == 1)[0])

        num_sel_links = len(selected_links)
        sorted_links = []
        node = np.where(b==1)[0].item()
        while num_sel_links != len(sorted_links):
            for link in selected_links:
                if A[node,link] == 1:
                    sorted_links.append(link)
                    node = np.where(A[:,link]==-1)[0].item()
                    selected_links.remove(link)
                    break
        sorted_links = [link+1 for link in sorted_links]

        cost = np.dot(sol.T,mu).item()

        return sorted_links, cost

def get_let_first_step(mu,A,b):
    sol = cvxopt_glpk_minmax(mu,A,b)
    selected_links = list(np.where(sol == 1)[0])

    node = np.where(b==1)[0].item()
    for link in selected_links:
        if A[node,link] == 1:
            first_step = link
            break

    cost = np.dot(sol.T,mu).item()

    return first_step, cost

def generate_cov_log(mu_ori, nu):
    n_link = np.size(mu_ori)
    
    sigma = nu*mu_ori*np.random.rand(n_link,1)

    sigma_log = np.log(np.divide(sigma**2,mu_ori**2)+1)
    mu_log = np.log(mu_ori)-0.5*sigma_log
    
    n_sample = n_link
    samples = np.zeros((n_link,n_sample))
    
    for i in range(np.shape(samples)[0]):
        samples[i] = np.random.lognormal(mu_log[i],np.sqrt(sigma_log[i]),(1,np.shape(samples)[1]))
    
    cov_ori = np.cov(samples)
    
    return cov_ori

def calc_logGP4_param(mu_ori, cov_ori):
    cov_log = np.log(cov_ori/np.dot(mu_ori,mu_ori.T)+1)
    mu_log = np.log(mu_ori)-0.5*np.diag(cov_log).reshape(-1,1)
    
    return mu_log, cov_log