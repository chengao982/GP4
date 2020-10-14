import numpy as np
import func
import time

def logGP4(A, A_idx, b, mu, sigma, r_0, r_s, N):
    A_k = A
    A_idx_k = A_idx
    b_k = b
    mu_k = mu
    sigma_k = sigma
    r_k = r_0

    total_cost = 0
    real_cost = []
    selected_links = []

    while r_k != r_s:
        # print('current node is %d' % (r_k+1))
        links_to_search = np.where(A_k[r_k,:]==1)
        value_min = float("inf")

        for link in links_to_search[0]:
            next_node = np.where(A_k[:,link]==-1)[0].item()

            if next_node == r_s:
                value = func.calc_exp_gauss(mu_k[link],sigma_k[link][link]).item()
                if value < value_min:
                    value_min = value
                    selected_link = link
                    selected_node = next_node

            else:
                v_hat = 0.0

                A_temp, b_temp = func.update_map(A_k,b_k,link,r_k,next_node)
                mu_sub, sigma_sub, sigma_con = func.update_param(mu_k,sigma_k,link)

                for i in range(0,N):
                    sample = np.random.normal(mu_sub[2], np.sqrt(sigma_sub[22]))
                    mu_con = func.update_mu(mu_sub,sigma_sub,sample)
                    mu_exp = func.calc_exp_gauss(mu_con,sigma_con)
                    x_temp = func.cvxopt_glpk_minmax(mu_exp,A_temp,b_temp)

                    if x_temp.all() == None:
                        v_hat = float("inf")
                        break
                    else:
                        v_hat = v_hat+np.dot(x_temp.T,mu_exp).item()

                value = func.calc_exp_gauss(mu_sub[2],sigma_sub[22]).item()+v_hat/N

                if value < value_min:
                    value_min = value
                    selected_link = link
                    selected_node = next_node

                    A_save = A_temp
                    b_save = b_temp
                    mu_sub_save = mu_sub
                    sigma_sub_save = sigma_sub
                    sigma_save = sigma_con

        selected_links.append(A_idx_k[selected_link])
        # print('Selected link is {}, whose value is {}'.format(selected_links[-1],value_min))

        cost = np.random.lognormal(mu_k[selected_link].item(), np.sqrt(sigma_k[selected_link,selected_link]))
        real_cost.append(cost)
        total_cost += cost
        # print('Sampled travel time is {}, running total cost is {}'.format(cost,total_cost))
        # print('-----------------------------------------------------------------------------------')

        r_k = selected_node

        if r_k != r_s:
            A_idx_k = np.delete(A_idx_k,selected_link)
            A_k = A_save
            b_k = b_save
            sigma_k = sigma_save
            mu_k = func.update_mu(mu_sub_save, sigma_sub_save, np.log(cost))
            
    return selected_links, real_cost, total_cost

def logGP4_iterations(A, A_idx, b, mu, sigma, r_0, r_s, N, iterations):
    results = []

    for ite in range(iterations):
        print('current iteration: %d' % ite)
        selected_links, real_cost, total_cost = logGP4(A, A_idx, b, mu, sigma, r_0, r_s, N)
        print('iteration finished, total cost is {}\nselected links are {}\ncorresponding cost are {}'.format(total_cost,selected_links,real_cost))
        print('************************************************************************************************')
        results.append(total_cost)

    average_result = np.sum(results)/iterations
    print('Average performance over {} iterations is {}'.format(iterations,average_result))

    return average_result

def log_dplus(A, A_idx, b, mu, sigma, r_0, r_s):
    A_k = A
    A_idx_k = A_idx
    b_k = b
    mu_k = mu
    sigma_k = sigma
    r_k = r_0

    total_cost = 0
    real_cost = []
    selected_links = []

    while r_k != r_s:
        # print('current node is %d' % (r_k+1))
        mu_exp = func.calc_exp_gauss(mu_k,sigma_k)
        link, value = func.get_let_first_step(mu_exp,A_k,b_k)
        next_node = np.where(A_k[:,link]==-1)[0].item()
        A_k, b_k = func.update_map(A_k,b_k,link,r_k,next_node)
        r_k = next_node
        mu_sub, sigma_sub, sigma_k = func.update_param(mu_k,sigma_k,link)
        cost = np.random.lognormal(mu_sub[2], np.sqrt(sigma_sub[22]))
        mu_k = func.update_mu(mu_sub,sigma_sub,np.log(cost))
        
        selected_links.append(A_idx_k[link])
        # print('Selected link is {}, whose value is {}'.format(selected_links[-1],value))
        A_idx_k = np.delete(A_idx_k,link)

        real_cost.append(cost)
        total_cost += cost
        # print('Sampled travel time is {}, running total cost is {}'.format(cost,total_cost))
        # print('-----------------------------------------------------------------------------------')

    return selected_links, real_cost, total_cost

def log_dplus_iterations(A, A_idx, b, mu, sigma, r_0, r_s, iterations):
    results = []

    for ite in range(iterations):
        print('current Dijkstra_plus iteration: %d' % ite)
        selected_links, real_cost, total_cost = log_dplus(A, A_idx, b, mu, sigma, r_0, r_s)
        print('iteration finished, total cost is {}\nselected links are {}\ncorresponding cost are {}'.format(total_cost,selected_links,real_cost))
        print('************************************************************************************************')
        results.append(total_cost)

    average_result = np.sum(results)/iterations
    print('Average performance over {} iterations is {}'.format(iterations,average_result))

    return average_result

# A, A_idx, b, r_0, r_s, n_link = func.generate_map(2)
# mu = func.generate_mu(n_link,1)
# sigma = func.generate_sigma(n_link,0.5)
# print(mu)
# print(sigma)

A, A_idx, mu_ori = func.extract_map(0)
cov_ori = func.generate_cov_log(mu_ori, 0.5)
mu, cov = func.calc_logGP4_param(mu_ori, cov_ori)
n_node = np.shape(A)[0]
origin = np.random.randint(0,int(0.5*n_node)) + 1
destination = np.random.randint(int(0.5*n_node),n_node) + 1
b, r_0, r_s = func.generate_b(n_node, 1, 15)

N = 100
iterations = 10

# log_dplus_iterations(A, A_idx, b, mu, cov, r_0, r_s, 10)
logGP4_iterations(A, A_idx, b, mu, cov, r_0, r_s, N, iterations)

dijkstra_selected_links, dijkstra_cost = func.get_let_path(mu_ori, A, b)
if dijkstra_cost is None:
    print('OD is not connected')
else:
    print('Dijkstra selected links are {}'.format(dijkstra_selected_links))
    print('Cost of Dijkstra optimal path is {}'.format(dijkstra_cost))

# n_test = 10
# test_results = [[],[],[]]
# OD_pairs = []

# for test in range(n_test):
#     origin = np.random.randint(0,int(0.5*n_node)) + 1
#     destination = np.random.randint(int(0.5*n_node),n_node) + 1
#     b, r_0, r_s = func.generate_b(n_node, origin, destination)
#     OD_pairs.append((r_0,r_s))
#     print('test # {}, O:{}, D:{}'.format(test,r_0,r_s))

#     dijkstra_selected_links, dijkstra_cost = func.get_let_path(mu_ori, A, b)
#     if dijkstra_cost is None:
#         print('OD is not connected')
#         continue
#     else:
#         test_results[0].append(dijkstra_cost)
#         print('Dijkstra selected links are {}'.format(dijkstra_selected_links))
#         print('Cost of Dijkstra optimal path is {}'.format(dijkstra_cost))

#     log_dplus_cost = log_dplus_iterations(A, A_idx, b, mu, cov, r_0, r_s, 10)
#     test_results[1].append(dijkstra_cost)

#     logGP4_cost = logGP4_iterations(A, A_idx, b, mu, cov, r_0, r_s, N, iterations)
#     test_results[2].append(logGP4_cost)

# print(test_results)
# print(OD_pairs)

# plt.figure()
# plt.plot(test_results[0], test_results[0], 'ko-')
# plt.plot(test_results[0], test_results[1], 'mo-')
# plt.plot(test_results[0], test_results[2], 'ro-')
# plt.show()
