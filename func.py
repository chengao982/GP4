import numpy as np
import cvxopt
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
    b = np.zeros((1,n_node))
    b[0][origin-1] = 1
    b[0][destination-1] = -1
    return b.T

def generate_mu(n_link, mu_scaler=10):
    mu = mu_scaler*np.ones((1,n_link))
    # mu[0][np.random.randint(1,n_link-1)] += 0.1
    mu[0][-1] = (n_link/2)*mu_scaler

    # mu = np.random.rand(1,n_link)
    # mu[0][-1] = n_link/4.5
    return mu.T

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
    b = generate_b(n_node,origin,destination)
    r_0 = origin-1
    r_s = destination-1
    return A, A_idx, b, r_0, r_s, n_link


# A, A_idx, b, r_0, r_s = generate_map(2)
# mu = generate_mu(3,2)
# sigma = generate_sigma(3)
# print(A)
# print(A_idx)
# print(b)
# print(mu)
# print(sigma)
# print(r_0)
# print(r_s)

def cvxopt_glpk_minmax(c, A, b, x_min=0):
    dim = np.size(c,0)

    x_min = x_min * np.ones(dim)
    # x_max = x_max * ones(n)
    # G = np.vstack([+np.eye(dim),-np.eye(dim)])
    # h = np.hstack([x_max, -x_min])
    G = -np.eye(dim)

    c = cvxopt.matrix(c,tc='d')
    A = cvxopt.matrix(A,tc='d')
    b = cvxopt.matrix(b,tc='d')
    G = cvxopt.matrix(G,tc='d')
    h = cvxopt.matrix(x_min.T,tc='d')
    sol = cvxopt.solvers.lp(c, G, h, A, b, solver='glpk',options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
    return np.array(sol['x'])

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
    sigma_12 = np.delete(sigma[:,link],link,axis=0).reshape((-1,1))
    sigma_21 = np.delete(sigma[link,:],link).reshape((1,-1))
    sigma_22 = sigma[link,link]
    sigma_sub = {11:sigma_11, 12:sigma_12, 21:sigma_21, 22:sigma_22}

    sigma_con = sigma_11-np.matmul(sigma_12,sigma_21)/sigma_22

    return mu_sub, sigma_sub, sigma_con

def update_mu(mu_sub, sigma_sub, sample):
    return mu_sub[1]+(sample-mu_sub[2])/sigma_sub[22]*sigma_sub[12]

def calc_exp_gauss(mu, sigma):
    sigma_diag = np.diag(sigma).reshape((-1,1)) if type(sigma) is np.ndarray else sigma
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

# array([[ 0.58783847, -0.08191563, -0.07014496,  0.04308853, -0.02114256, -0.09272821],
#        [-0.08191563,  0.60522864,  0.01545851, -0.09300809,  0.22951716,  0.04012107],
#        [-0.07014496,  0.01545851,  0.63474145, -0.03136713,  0.15563712, -0.0023954 ],
#        [ 0.04308853, -0.09300809, -0.03136713,  0.55790984,  0.0241525 , -0.09513229],
#        [-0.02114256,  0.22951716,  0.15563712,  0.0241525 ,  0.37843407,  0.2149459 ],
#        [-0.09272821,  0.04012107, -0.0023954 , -0.09513229,  0.2149459 ,  0.57376631]])