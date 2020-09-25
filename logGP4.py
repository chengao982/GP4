import numpy as np
import cvxopt
import math
import func

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
        print('current node is %d' % r_k)
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
                    sample = np.random.normal(mu_sub[2], math.sqrt(sigma_sub[22]))
                    mu_con = func.update_mu(mu_sub,sigma_sub,sample)
                    mu_exp = func.calc_exp_gauss(mu_con,sigma_con)
                    x_temp = func.cvxopt_glpk_minmax(mu_exp,A_temp,b_temp)

                    v_hat = v_hat+np.dot(x_temp.T,mu_exp).item()

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
        print('Selected link is {}, whose value is {}'.format(selected_links[-1],value_min))

        cost = np.random.lognormal(mu_k[selected_link].item(), math.sqrt(sigma_k[selected_link,selected_link]))
        real_cost.append(cost)
        total_cost += cost
        print('Sampled travel time is {}, running total cost is {}'.format(cost,total_cost))
        print('-----------------------------------------------------------------------------------')

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

    for ite in range(0,iterations):
        print('current iteration: %d' % ite)
        selected_links, real_cost, total_cost = logGP4(A, A_idx, b, mu, sigma, r_0, r_s, N)
        print('iteration finished, total cost is {}\nselected_links are {}\ncorresponding cost are {}'.format(total_cost,selected_links,real_cost))
        print('************************************************************************************************')
        results.append(total_cost)

    average_result = np.sum(results)/iterations
    print('Average performance over {} iterations is {}'.format(iterations,average_result))

    return average_result


A, A_idx, b, r_0, r_s, n_link = func.generate_map(2)
mu = func.generate_mu(n_link,1)
sigma = func.generate_sigma(n_link,0.5)
print(mu)
print(sigma)

N = 100
iterations = 10

logGP4_iterations(A, A_idx, b, mu, sigma, r_0, r_s, N, iterations)