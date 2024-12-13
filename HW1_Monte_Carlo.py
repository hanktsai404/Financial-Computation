'''
Financial Computation: Assignment 1 (bonus)
Author: Cheng-Han Tsai
b07703014@ntu.edu.tw
'''

import FINCOMP.MM_Simulator as MM_sim
import math
import numpy as np


# Input
S_0 = 100
r = 0.05
q = 0.02
sigma = 0.5
T = 0.4
K_1 = 90
K_2 = 98
K_3 = 102
K_4 = 104
sample_size = 10000
ite = 20
N = sample_size * ite # Sample size

MM_stock = MM_sim.stock_price_simulator(S_0, float(0), r, q, sigma)
price_list = []

for i in range(0, ite):
    MM_stock.simulate_price(T, sample_size, 1)
    payoffs = list()
    for price in MM_stock.sim_price:
        if price<K_1 or price>=K_4:
            payoffs.append(0)
        elif price>=K_1 and price<K_2:
            payoffs.append(price-K_1)
        elif price>=K_2 and price<K_3:
            payoffs.append(K_2-K_1)
        else:
            payoffs.append((K_2-K_1)/(K_4-K_3)*(K_4-price))
    deriv_price = math.exp(0-(r*T)) * sum(payoffs)/sample_size
    price_list.append(deriv_price)

mean = np.mean(price_list)
std = np.std(price_list)
lower_bound = mean - (2*std)
upper_bound = mean + (2*std)

print("The price CI of the given derivative is (by Monte-Carlo simulation):", end="\n")
print("[", lower_bound, upper_bound, "]")