'''
Financial Computation: Assignment 4
Author: Cheng-Han Tsai
b07703014@ntu.edu.tw
'''

from FINCOMP import MM_Simulator as MM_sim
from FINCOMP import BinTree
import numpy as np

# Monte Carlo parameters
SAMPLE_SIZE = 1000
ITE = 20

# Global parameters 
S_t = 50
r = 0.1
q = 0
sigma = 0.4
t = 0.25
T = 0.5
n_period = 100

def basic_and_bonus1(S_t, r, q, sigma, S_max_t, t, T, n_period):
    # Binomial Tree
    print("Binomial Tree:")
    bin_tree = BinTree.path_dependent_CRR_tree(S_t, r, q, sigma, T, n_period, t, S_max_t)
    bin_tree.grow_maxmin_tree(is_max=True)
    eu_price = bin_tree.floating_lb_backward_induction(is_call=False, is_eu=True)
    am_price = bin_tree.floating_lb_backward_induction(is_call=False, is_eu=False)
    print("European put:", eu_price)
    print("American put:", am_price)
    print()

    # Monte-Carlo
    print("Monte-Carlo method (95% C.I.) for European put:")
    lb_sim = MM_sim.path_max_simulator(S_t, 0, r, q, sigma)
    lb_sim.simulate_floating_lb_puts(T, t, S_max_t, n_period, SAMPLE_SIZE, ITE)
    lb_sim.floating_lb_put_pricing(T, t)
    put_mean = np.mean(lb_sim.deriv_prices)
    put_std = np.std(lb_sim.deriv_prices)
    put_upper = put_mean + (2*put_std)
    put_lower = put_mean - (2*put_std)
    print("[", put_lower, ",", put_upper, "]")
    print()

    # Fast Binomial Tree (Bonus 1)
    print("Growing binomial tree by determining S_max directly:")
    bin_tree.grow_maxmin_tree_fast(is_max=True)
    eu_price = bin_tree.floating_lb_backward_induction(is_call=False, is_eu=True)
    am_price = bin_tree.floating_lb_backward_induction(is_call=False, is_eu=False)
    print("European put:", eu_price)
    print("American put:", am_price)
    print()



S_max_t = 50
print("S_max_t = 50")
basic_and_bonus1(S_t, r, q, sigma, S_max_t, t, T, n_period)
print()

S_max_t = 60
print("S_max_t = 60")
basic_and_bonus1(S_t, r, q, sigma, S_max_t, t, T, n_period)
print()

S_max_t = 70
print("S_max_t = 70")
basic_and_bonus1(S_t, r, q, sigma, S_max_t, t, T, n_period)
print()

# Cheuk and Vorst (1997) (Bonus 2)
n_period = 1000
print("Cheuk and Vorst (1997):")
eu_price = BinTree.CV_floating_lb_put(S_t, r, q, sigma, T, n_period, t, True)
am_price = BinTree.CV_floating_lb_put(S_t, r, q, sigma, T, n_period, t, False)
print("European put:", eu_price)
print("American put:", am_price)
print()
# This method is faster and requires less memory space