'''
Financial Computation: Assignment 2
Author: Cheng-Han Tsai
b07703014@ntu.edu.tw
'''

from FINCOMP import MM_Simulator as MM_sim
from FINCOMP import Close_Form, Payoffs, BinTree
import numpy as np

S_0 = 140
r = 0.04
q = 0
sigma = 0.25
T = 0.5
K = 150


# Black-Scholes
print("Black-Scholes Formula", end="\n")
print("Call option:\t", Close_Form.eu_call_price(S_0, K, r, q, sigma, T), end="\n")
print("Put option:\t", Close_Form.eu_put_price(S_0, K, r, q, sigma, T), end="\n")
print("\n")

# Monte-Carlo (Call and Put use the same price sample)
sample_size = 10000
ite = 20

print("Monte Carlo Simulation", end="\n")
MM_stock = MM_sim.stock_price_simulator(S_0, 0, r, q, sigma)
MM_stock.simulate_price(T, sample_size, ite)

# Call
MM_stock.deriv_pricing(T, lambda x: Payoffs.eu_call_payoff(x, K))
call_mean = np.mean(MM_stock.deriv_prices)
call_std = np.std(MM_stock.deriv_prices)
call_lower_bound = call_mean - (2*call_std)
call_upper_bound = call_mean + (2*call_std)
print("95% C.I. for call option: [", call_lower_bound, ",", call_upper_bound, "]", end="\n")

# Put
MM_stock.deriv_pricing(T, lambda x: Payoffs.eu_put_payoff(x, K))
put_mean = np.mean(MM_stock.deriv_prices)
put_std = np.std(MM_stock.deriv_prices)
put_lower_bound = put_mean - (2*put_std)
put_upper_bound = put_mean + (2*put_std)
print("95% C.I. for put option: [", put_lower_bound, ",", put_upper_bound, "]", end="\n")

print()


# CRR binomial tree 2d
n_period = 2

print("CRR Binomial Tree (2d array)", end="\n")
stock_tree = BinTree.CRR_tree(S_0, r, q, sigma, T, n_period)
stock_tree.grow_simple_tree()
print("u = ", stock_tree.u, ", d = ", stock_tree.d, ", p = ", stock_tree.prob, end = "\n")

backward_inductor = BinTree.backward_inductor_2d(stock_tree)
# European
print("European Call Option:\t", backward_inductor.backward(lambda x: Payoffs.eu_call_payoff(x, K)))
print("European Put Option:\t", backward_inductor.backward(lambda x: Payoffs.eu_put_payoff(x, K)))

# American
print("American Call Option:\t", backward_inductor.backward(lambda x: Payoffs.eu_call_payoff(x, K), is_eu=False))
print("American Call Option:\t", backward_inductor.backward(lambda x: Payoffs.eu_put_payoff(x, K), is_eu=False))

print()

# Bonus
# CRR Binomial Tree 1d
print("CRR Binomial Tree (1d array)",end="\n")
backward_inductor = BinTree.backward_inductor_1d(stock_tree)
# European
print("European Call Option:\t", backward_inductor.backward(lambda x: Payoffs.eu_call_payoff(x, K)))
print("European Put Option:\t", backward_inductor.backward(lambda x: Payoffs.eu_put_payoff(x, K)))

# American
print("American Call Option:\t", backward_inductor.backward(lambda x: Payoffs.eu_call_payoff(x, K), is_eu=False))
print("American Call Option:\t", backward_inductor.backward(lambda x: Payoffs.eu_put_payoff(x, K), is_eu=False))
print()

# Combinatorial method
print("Combinatorial Method")
print("European Call Option:\t", BinTree.eu_comb_method(S_0, r, q, sigma, T, n_period, lambda x: Payoffs.eu_call_payoff(x, K)))
print("European Put Option:\t", BinTree.eu_comb_method(S_0, r, q, sigma, T, n_period, lambda x: Payoffs.eu_put_payoff(x, K)))