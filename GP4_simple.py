import numpy as np
import func
import matplotlib.pyplot as plt
import time
import plot
import os
import pandas as pd

def GP4(A, A_idx, b, mu, sigma, r_0, r_s, N):
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
                value = mu_k[link].item()
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
                    x_temp = func.cvxopt_glpk_minmax(mu_con,A_temp,b_temp)

                    if x_temp.all() == None:
                        v_hat = float("inf")
                        break
                    else:
                        v_hat = v_hat+np.dot(x_temp.T,mu_con).item()

                value = mu_sub[2]+v_hat/N

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

        cost = np.random.normal(mu_k[selected_link].item(), np.sqrt(sigma_k[selected_link,selected_link]))
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
            mu_k = func.update_mu(mu_sub_save, sigma_sub_save, cost)

    return selected_links, real_cost, total_cost

def GP4_iterations(A, A_idx, b, mu, sigma, r_0, r_s, N, iterations):
    results = []

    for ite in range(iterations):
        # print('current GP4 iteration: %d' % ite)
        selected_links, real_cost, total_cost = GP4(A, A_idx, b, mu, sigma, r_0, r_s, N)
        # print('iteration finished, total cost is {}\nselected links are {}\ncorresponding cost are {}'.format(total_cost,selected_links,real_cost))
        # print('selected links are {}'.format(selected_links))
        # print('************************************************************************************************')
        results.append(total_cost)

    average_result = np.sum(results)/iterations
    print('GP4: Average performance over {} iterations is {}'.format(iterations,average_result))

    return average_result

def dplus(A, A_idx, b, mu, sigma, r_0, r_s):
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
        link, value = func.get_let_first_step(mu_k,A_k,b_k)
        next_node = np.where(A_k[:,link]==-1)[0].item()
        A_k, b_k = func.update_map(A_k,b_k,link,r_k,next_node)
        r_k = next_node
        mu_sub, sigma_sub, sigma_k = func.update_param(mu_k,sigma_k,link)
        cost = np.random.normal(mu_sub[2], np.sqrt(sigma_sub[22]))
        mu_k = func.update_mu(mu_sub,sigma_sub,cost)
        
        selected_links.append(A_idx_k[link])
        # print('Selected link is {}, whose value is {}'.format(selected_links[-1],value))
        A_idx_k = np.delete(A_idx_k,link)

        real_cost.append(cost)
        total_cost += cost
        # print('Sampled travel time is {}, running total cost is {}'.format(cost,total_cost))
        # print('-----------------------------------------------------------------------------------')

    return selected_links, real_cost, total_cost

def dplus_iterations(A, A_idx, b, mu, sigma, r_0, r_s, iterations):
    results = []

    for ite in range(iterations):
        # print('current Dijkstra_plus iteration: %d' % ite)
        selected_links, real_cost, total_cost = dplus(A, A_idx, b, mu, sigma, r_0, r_s)
        print('iteration finished, total cost is {}\nselected links are {}\ncorresponding cost are {}'.format(total_cost,selected_links,real_cost))
        print('************************************************************************************************')
        results.append(total_cost)

    average_result = np.sum(results)/iterations
    print('D+: Average performance over {} iterations is {}'.format(iterations,average_result))

    return average_result


N = 100
iterations = 100

n_test = 30
test_results = [[],[],[]]
OD_pairs = []

for test in range(1):
    A, A_idx, b, r_0, r_s, n_link = func.generate_map(10)
    mu = func.generate_mu(n_link)
    cov = func.generate_cov(mu, 0.5)

    OD_pairs.append((r_0,r_s))
    print('test # {}, O:{}, D:{}'.format(test,r_0,r_s))

    dijkstra_selected_links, dijkstra_cost = func.get_let_path(mu, A, b)
    if dijkstra_cost is None:
        print('OD is not connected')
        continue
    else:
        test_results[0].append(dijkstra_cost)
        print('Dijkstra selected links are {}'.format(dijkstra_selected_links))
        print('Cost of Dijkstra optimal path is {}'.format(dijkstra_cost))

    # dplus_cost = dplus_iterations(A, A_idx, b, mu, cov, r_0, r_s, iterations)
    # test_results[1].append(dplus_cost)

    GP4_cost = GP4_iterations(A, A_idx, b, mu, cov, r_0, r_s, N, 1)
    test_results[2].append(GP4_cost)


# plot.GP4_plot(test_results,OD_pairs)

print(test_results)
print(OD_pairs)
