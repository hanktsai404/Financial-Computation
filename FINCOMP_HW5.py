'''
Financial Computation: Assignment 5
Author: Cheng-Han Tsai
b07703014@ntu.edu.tw
'''

from FINCOMP import MM_Simulator as MM_sim
from FINCOMP import BinTree, Payoffs
import numpy as np
import matplotlib.pyplot as plt
import time

# Monte Carlo parameters
SAMPLE_SIZE = 1000
ITE = 20

# Binomial tree parameters
M = 100

# Global parameters
S_t = 50
S_ave_t = 50
r = 0.1
q = 0.05
sigma = 0.8
K = 50
n_period = 100

'''Basic Requirements and bonus 2'''
# 1. t=0, T=0.25
t = 0
T = 0.25
print("t=", t, "\t", "T=", T)
# Binomial Tree
# We only perform bonus 2 in this case to save time latter
print("Binomial Tree method (linear):")
bin_tree = BinTree.path_dependent_CRR_tree(S_t, r, q, sigma, T, n_period, t, S_ave_t)
bin_tree.grow_arith_ave_tree(M, is_linear=True)
start = time.time()
eu_price = bin_tree.asian_backward_induction(True, lambda x: Payoffs.eu_call_payoff(x, K))
end = time.time()
print("European call (linear interpolation search):", eu_price, "\t", end-start, "s")
start=time.time()
am_price = bin_tree.asian_backward_induction(False, lambda x: Payoffs.eu_call_payoff(x, K))
end = time.time()
print("American call (linear interpolation search):", am_price, "\t", end-start, "s")
print()

start = time.time()
eu_price = bin_tree.asian_backward_induction(True, lambda x: Payoffs.eu_call_payoff(x, K), is_bin_search=True)
end = time.time()
print("European call (binary search):", eu_price, "\t", end-start, "s")
start=time.time()
am_price = bin_tree.asian_backward_induction(False, lambda x: Payoffs.eu_call_payoff(x, K), is_bin_search=True)
end = time.time()
print("American call (binary search):", am_price, "\t", end-start, "s")
print()

start = time.time()
eu_price = bin_tree.asian_backward_induction(True, lambda x: Payoffs.eu_call_payoff(x, K), is_seq_search=True)
end = time.time()
print("European call (sequential search):", eu_price, "\t", end-start, "s")
start=time.time()
am_price = bin_tree.asian_backward_induction(False, lambda x: Payoffs.eu_call_payoff(x, K), is_seq_search=True)
end = time.time()
print("American call (sequential search):", am_price, "\t", end-start, "s")
print()


print("Binomial Tree method (log):")
bin_tree.grow_arith_ave_tree(M, is_linear=False)
eu_price = bin_tree.asian_backward_induction(True, lambda x: Payoffs.eu_call_payoff(x, K))
am_price = bin_tree.asian_backward_induction(False, lambda x: Payoffs.eu_call_payoff(x, K))
eu_price = bin_tree.asian_backward_induction(True, lambda x: Payoffs.eu_call_payoff(x, K))
am_price = bin_tree.asian_backward_induction(False, lambda x: Payoffs.eu_call_payoff(x, K))
print("European call:", eu_price, "\t", end-start)
print("American call:", am_price, "\t", end-start)
print()

# Monte-Carlo
print("Monte-Carlo method (95% C.I.) for European call:")
ave_sim = MM_sim.path_arith_ave_simulator(S_t, 0, r, q, sigma)
ave_sim.simulate_ave_deriv_values(T, t, S_ave_t, n_period, SAMPLE_SIZE, ITE, lambda x: Payoffs.eu_call_payoff(x, K))
ave_sim.ave_deriv_pricing(T, t)
call_mean = np.mean(ave_sim.deriv_prices)
call_std = np.std(ave_sim.deriv_prices)
call_upper = call_mean + (2*call_std)
call_lower = call_mean - (2*call_std)
print("[", call_lower, ",", call_upper, "]")
print()


# 2. t=0.25, T=0.5
t = 0.25
T = 0.5
print("t=", t, "\t", "T=", T)
# Binomial Tree
print("Binomial Tree method (linear):")
bin_tree = BinTree.path_dependent_CRR_tree(S_t, r, q, sigma, T, n_period, t, S_ave_t)
bin_tree.grow_arith_ave_tree(M, is_linear=True)
eu_price = bin_tree.asian_backward_induction(True, lambda x: Payoffs.eu_call_payoff(x, K))
am_price = bin_tree.asian_backward_induction(False, lambda x: Payoffs.eu_call_payoff(x, K))
print("European call:", eu_price)
print("American call:", am_price)
print()

print("Binomial Tree method (log):")
bin_tree.grow_arith_ave_tree(M, is_linear=False)
eu_price = bin_tree.asian_backward_induction(True, lambda x: Payoffs.eu_call_payoff(x, K))
am_price = bin_tree.asian_backward_induction(False, lambda x: Payoffs.eu_call_payoff(x, K))
print("European call:", eu_price)
print("American call:", am_price)
print()

# Monte-Carlo
print("Monte-Carlo method (95% C.I.) for European call:")
ave_sim = MM_sim.path_arith_ave_simulator(S_t, 0, r, q, sigma)
ave_sim.simulate_ave_deriv_values(T, t, S_ave_t, n_period, SAMPLE_SIZE, ITE, lambda x: Payoffs.eu_call_payoff(x, K))
ave_sim.ave_deriv_pricing(T, t)
call_mean = np.mean(ave_sim.deriv_prices)
call_std = np.std(ave_sim.deriv_prices)
call_upper = call_mean + (2*call_std)
call_lower = call_mean - (2*call_std)
print("[", call_lower, ",", call_upper, "]")
print()

'''Bonus 1'''
# We use t=0.25, T=0.5 for demonstation
t = 0.25
T = 0.5
bin_tree = BinTree.path_dependent_CRR_tree(S_t, r, q, sigma, T, n_period, t, S_ave_t)

M_grid = [i for i in range(50, 401) if i%50==0] # M=50, 100, 150, 200, ..., 400
linear_eu_values = []
log_eu_values = []
linear_am_values = []
log_am_values = []

print("\tLinear\t\t\t\tLog")
print("M\tEuropean\tAmerican\tEuropean\tAmerican")

for m in M_grid:
    bin_tree.grow_arith_ave_tree(m, is_linear=True)
    linear_eu = bin_tree.asian_backward_induction(True, lambda x: Payoffs.eu_call_payoff(x, K))
    linear_am = bin_tree.asian_backward_induction(False, lambda x: Payoffs.eu_call_payoff(x, K))
    linear_eu_values.append(linear_eu)
    linear_am_values.append(linear_am)

    bin_tree.grow_arith_ave_tree(m, is_linear=False)
    log_eu = bin_tree.asian_backward_induction(True, lambda x: Payoffs.eu_call_payoff(x, K))
    log_am = bin_tree.asian_backward_induction(False, lambda x: Payoffs.eu_call_payoff(x, K))
    log_eu_values.append(log_eu)
    log_am_values.append(log_am)

    print(m, "\t", round(linear_eu, 4), "\t", round(linear_am, 4), "\t", round(log_eu, 4), "\t", round(log_am, 4))


plt.figure()
plt.plot(M_grid, linear_eu_values, label="Linear")
plt.plot(M_grid, log_eu_values, label="Log")
plt.xlabel("M")
plt.ylabel("Option Price (binomial tree)")
plt.title("European Option")
plt.legend()
plt.show()

plt.figure()
plt.plot(M_grid, linear_am_values, label="Linear")
plt.plot(M_grid, log_am_values, label="Log")
plt.xlabel("M")
plt.ylabel("Option Price (binomial tree)")
plt.title("American Option")
plt.legend()
plt.show()
