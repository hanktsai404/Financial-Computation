'''
Financial Computation: Assignment 3
Author: Cheng-Han Tsai
b07703014@ntu.edu.tw
'''

from FINCOMP import MM_Simulator, Payoffs
import numpy as np
import time

ITE = 20
SAMPLE_SIZE = 10000
r = 0.1
T = 0.5
k = 100

def present_assignment(s_0_arr, q_arr, sigma_arr, corr_matrix):
    n_stocks = np.shape(s_0_arr)[0]
    cov_matrix = np.zeros([n_stocks, n_stocks])
    for i in range(0, n_stocks):
        for j in range(0, n_stocks):
            cov_matrix[i,j] = sigma_arr[i]*sigma_arr[j]*corr_matrix[i,j]*T

    MM_multi = MM_Simulator.multi_stock_price_simulator(s_0_arr, None, r, q_arr, sigma_arr, cov_matrix)

    # Common approach
    time_start = time.time()
    MM_multi.simulate_prices(T, SAMPLE_SIZE, ITE)
    call_prices = MM_multi.deriv_pricing(T, lambda x: Payoffs.eu_rainbow_call_payoff(x, k))
    call_mean = np.mean(call_prices)
    call_std = np.std(call_prices)
    call_upper = call_mean + (2*call_std)
    call_lower = call_mean - (2*call_std)
    time_end = time.time()
    cost = time_end-time_start

    print("Monte-Carlo Simulation of Rainbow Call (95% C.I.):", end="\n")
    print("[", call_lower, ",", call_upper, "]")
    print("Time Cost:", cost, "s")
    print()

    # Antithetic variate and moment matching
    time_start = time.time()
    MM_multi.simulate_prices(T, SAMPLE_SIZE, ITE, is_anti_moment=True)
    call_prices = MM_multi.deriv_pricing(T, lambda x: Payoffs.eu_rainbow_call_payoff(x, k))
    call_mean = np.mean(call_prices)
    call_std = np.std(call_prices)
    call_upper = call_mean + (2*call_std)
    call_lower = call_mean - (2*call_std)
    time_end = time.time()
    cost = time_end-time_start

    print("Monte-Carlo Simulation of Rainbow Call (Antithetic Variate and Moment Matching, 95% C.I.):", end="\n")
    print("[", call_lower, ",", call_upper, "]")
    print("Time Cost:", cost, "s")
    print()

    # Inverse Cholesky Method proposed by Wang (2008)
    time_start = time.time()
    MM_multi.simulate_prices(T, SAMPLE_SIZE, ITE, is_inv_cholesky=True)
    call_prices = MM_multi.deriv_pricing(T, lambda x: Payoffs.eu_rainbow_call_payoff(x, k))
    call_mean = np.mean(call_prices)
    call_std = np.std(call_prices)
    call_upper = call_mean + (2*call_std)
    call_lower = call_mean - (2*call_std)
    time_end = time.time()
    cost = time_end-time_start

    print("Monte-Carlo Simulation of Rainbow Call (Inverse Cholesky Method, 95% C.I.):", end="\n")
    print("[", call_lower, ",", call_upper, "]")
    print("Time Cost:", cost, "s")
    print()

'''(i)'''
s_0_arr = np.array([95, 95])
q_arr = np.array([0.05, 0.05])
sigma_arr = np.array([0.5, 0.5])
corr_matrix = np.array([
    [1, 1],
    [1, 1],
]) # Enter correlation first

print("(i)", end="\n")
present_assignment(s_0_arr, q_arr, sigma_arr, corr_matrix)

'''(ii)'''
s_0_arr = np.array([95, 95])
q_arr = np.array([0.05, 0.05])
sigma_arr = np.array([0.5, 0.5])
corr_matrix = np.array([
    [1, -1],
    [-1, 1],
]) # Enter correlation first

print("(ii)", end="\n")
present_assignment(s_0_arr, q_arr, sigma_arr, corr_matrix)

'''(iii)'''
s_0_arr = np.array([95, 95, 95, 95, 95])
q_arr = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
sigma_arr = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
corr_matrix = np.array([
    [1, 0.5, 0.5, 0.5, 0.5],
    [0.5, 1, 0.5, 0.5, 0.5],
    [0.5, 0.5 ,1, 0.5, 0.5],
    [0.5, 0.5, 0.5, 1, 0.5],
    [0.5, 0.5, 0.5, 0.5, 1]
]) # Enter correlation first

print("(iii)", end="\n")
present_assignment(s_0_arr, q_arr, sigma_arr, corr_matrix)