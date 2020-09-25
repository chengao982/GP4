import numpy as np
import cvxopt
import math
import func
import time

def BiGP4(A, A_idx, b, phi, mu1, sigma1, mu2, sigma2, r_0, r_s, N):
    A_k = A
    A_idx_k = A_idx
    b_k = b
    mu1_k = mu1
    sigma1_k = sigma1
    mu2_k = mu2
    sigma2_k = sigma2
    r_k = r_0

    total_cost = 0
    real_cost = []
    selected_links = []

    while r_k != r_s:
        print('current node is %d' % r_k)
        links_to_search = np.where(A_k[r_k,:]==1)
        value_min = float("inf")

        for link in links_to_search[0]:
            next_node = np.where(A_k[:,link]==-1)[0].item()

            if next_node == r_s:
                value = func.calc_bi_gauss(phi,mu1_k[link],mu2_k[link]).item()
                if value < value_min:
                    value_min = value
                    selected_link = link
                    selected_node = next_node

            else:
                v_hat = 0.0

                A_temp, b_temp = func.update_map(A_k,b_k,link,r_k,next_node)
                mu1_sub, sigma1_sub, sigma1_con = func.update_param(mu1_k,sigma1_k,link)
                mu2_sub, sigma2_sub, sigma2_con = func.update_param(mu2_k,sigma2_k,link)

                for i in range(0,N):
                    sample1 = np.random.normal(mu1_sub[2], math.sqrt(sigma1_sub[22]))
                    sample2 = np.random.normal(mu2_sub[2], math.sqrt(sigma2_sub[22]))
                    sample = func.calc_bi_gauss(phi,sample1,sample2)
                    mu1_con = func.update_mu(mu1_sub,sigma1_sub,sample)
                    mu2_con = func.update_mu(mu2_sub,sigma2_sub,sample)
                    mu_con = func.calc_bi_gauss(phi,mu1_con,mu2_con)
                    x_temp = func.cvxopt_glpk_minmax(mu_con,A_temp,b_temp)

                    v_hat = v_hat+np.dot(x_temp.T,mu_con).item()

                value = func.calc_bi_gauss(phi,mu1_sub[2],mu2_sub[2])+v_hat/N

                if value < value_min:
                    value_min = value
                    selected_link = link
                    selected_node = next_node

                    A_save = A_temp
                    b_save = b_temp
                    mu1_sub_save = mu1_sub
                    sigma1_sub_save = sigma1_sub
                    sigma1_save = sigma1_con
                    mu2_sub_save = mu2_sub
                    sigma2_sub_save = sigma2_sub
                    sigma2_save = sigma2_con

        selected_links.append(A_idx_k[selected_link])
        print('Selected link is {}, whose value is {}'.format(selected_links[-1],value_min))

        cost1 = np.random.normal(mu1_k[selected_link], math.sqrt(sigma1_k[selected_link,selected_link])).item()
        cost2 = np.random.normal(mu2_k[selected_link], math.sqrt(sigma2_k[selected_link,selected_link])).item()
        cost = func.calc_bi_gauss(phi,cost1,cost2)
        real_cost.append(cost)
        total_cost += cost
        print('Sampled travel time is {}, running total cost is {}'.format(cost,total_cost))
        print('--------------------------------------------------------------')

        r_k = selected_node

        if r_k != r_s:
            A_idx_k = np.delete(A_idx_k,selected_link)
            A_k = A_save
            b_k = b_save
            sigma1_k = sigma1_save
            sigma2_k = sigma2_save
            mu1_k = func.update_mu(mu1_sub_save, sigma1_sub_save, cost)
            mu2_k = func.update_mu(mu2_sub_save, sigma2_sub_save, cost)

    return selected_links, real_cost, total_cost

def BiGP4_iterations(A, A_idx, b, phi, mu1, sigma1, mu2, sigma2, r_0, r_s, N, iterations):
    results = []

    for ite in range(0,iterations):
        print('current iteration: %d' % ite)
        selected_links, real_cost, total_cost = BiGP4(A, A_idx, b, phi, mu1, sigma1, mu2, sigma2, r_0, r_s, N)
        print('iteration finished, total cost is {}\nselected_links are {}\ncorresponding cost are {}'.format(total_cost,selected_links,real_cost))
        print('********************************************************************')
        results.append(total_cost)

    average_result = np.sum(results)/iterations
    print('Average performance over {} iterations is {}'.format(iterations,average_result))

    return average_result


A, A_idx, b, r_0, r_s, n_link = func.generate_map(2)
phi = np.random.rand()
mu1 = func.generate_mu(n_link,10.5)
sigma1 = func.generate_sigma(n_link)
mu2 = func.generate_mu(n_link,9.5)
sigma2 = func.generate_sigma(n_link)

print(phi)
print(mu1)
print(sigma1)
print(mu2)
print(sigma2)

N = 100
iterations = 10

BiGP4_iterations(A, A_idx, b, phi, mu1, sigma1, mu2, sigma2, r_0, r_s, N, iterations)