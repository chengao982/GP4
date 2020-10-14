import numpy as np
import func
import matplotlib.pyplot as plt
import time
import plot

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
        print('iteration finished, total cost is {}\nselected links are {}'.format(total_cost,selected_links))
        # print('************************************************************************************************')
        results.append(total_cost)

    average_result = np.sum(results)/iterations
    print('Average performance over {} iterations is {}'.format(iterations,average_result))
    print('************************************************************************************************')

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
        # print('iteration finished, total cost is {}\nselected links are {}\ncorresponding cost are {}'.format(total_cost,selected_links,real_cost))
        # print('************************************************************************************************')
        results.append(total_cost)

    average_result = np.sum(results)/iterations
    print('Average performance over {} iterations is {}'.format(iterations,average_result))
    print('************************************************************************************************')

    return average_result

def dijkstra(A, b, mu, sigma):
    mu_k = mu
    sigma_k = sigma
    r_k = r_0

    total_cost = 0
    real_cost = []

    selected_links, dijkstra_cost = func.get_let_path(mu, A, b)
    num_sel_links = len(selected_links)
    links = np.array(selected_links)-1

    for i in range(num_sel_links):
        mu_sub, sigma_sub, sigma_k = func.update_param(mu_k,sigma_k,links[0])
        cost = np.random.normal(mu_sub[2], np.sqrt(sigma_sub[22]))
        real_cost.append(cost)
        total_cost += cost
        mu_k = func.update_mu(mu_sub,sigma_sub,cost)
        for j in range(1,len(links)):
            if links[j] > links[0]:
                links[j] -= 1
        links = np.delete(links,0)

    return selected_links, real_cost, total_cost

def dijkstra_iterations(A, b, mu, sigma, iterations):
    results = []

    for ite in range(iterations):
        # print('current Dijkstra iteration: %d' % ite)
        selected_links, real_cost, total_cost = dijkstra(A, b, mu, sigma)
        # print('iteration finished, total cost is {}\ncorresponding cost are {}'.format(total_cost,real_cost))
        # print('************************************************************************************************')
        results.append(total_cost)

    average_result = np.sum(results)/iterations
    print('Average performance over {} iterations is {}'.format(iterations,average_result))
    print('************************************************************************************************')

    return average_result

# start = time.process_time()

# A, A_idx, b, r_0, r_s, n_link = func.generate_map(2)
# mu = func.generate_mu(n_link)
# sigma = func.generate_sigma(n_link,3)

A, A_idx, mu = func.extract_map(0)
_, sig, cov1 = func.generate_cov1(mu, 0.5, 50)
corr, _, _ = func.generate_cov1(mu, 0.5, 10)
cov2 = sig*corr
cov3 = sig*np.eye(np.size(mu))
n_node = np.shape(A)[0]
# origin = np.random.randint(0,int(0.5*n_node))
# destination = np.random.randint(int(0.5*n_node),n_node)
# b, r_0, r_s = func.generate_b(n_node, origin, destination)

N = 100
iterations = 200

n_test = 10
test_results = [[],[],[]]
OD_pairs = []

for test in range(n_test):
    origin = np.random.randint(0,int(0.5*n_node)) + 1
    destination = np.random.randint(int(0.5*n_node),n_node) + 1
    b, r_0, r_s = func.generate_b(n_node, origin, destination)
    OD_pairs.append((r_0,r_s))                                  #+1 to get true index of node
    print('test # {}, O:{}, D:{}'.format(test,r_0,r_s))

    dijkstra_selected_links, dijkstra_cost = func.get_let_path(mu, A, b)
    if dijkstra_cost is None:
        print('OD is not connected')
        continue
    else:
        test_results[0].append(dijkstra_cost)
        print('Dijkstra selected links are {}'.format(dijkstra_selected_links))
        print('Cost of Dijkstra optimal path is {}'.format(dijkstra_cost))
    print('************************************************************************************************')

    dijkstra_cost = dijkstra_iterations(A, b, mu, cov2, iterations)
    test_results[1].append(dijkstra_cost)

    # dplus_cost = dplus_iterations(A, A_idx, b, mu, cov, r_0, r_s, iterations)
    # test_results[1].append(dplus_cost)

    # GP4_cost = GP4_iterations(A, A_idx, b, mu, cov1, r_0, r_s, N, iterations)
    # test_results[1].append(GP4_cost)

    GP4_cost = GP4_iterations(A, A_idx, b, mu, cov2, r_0, r_s, N, iterations)
    test_results[2].append(GP4_cost)

# end = time.process_time()
# print(end-start)

print(test_results)
print(OD_pairs)

# plot.GP4_plot(test_results,OD_pairs)
