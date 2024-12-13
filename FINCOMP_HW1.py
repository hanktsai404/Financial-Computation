'''
Financial Computation: Assignment 1
Author: Cheng-Han Tsai
b07703014@ntu.edu.tw
'''

from scipy.stats import norm
import math

'''-----------------------------------Functions------------------------------------'''
def d_1(s_0, k, r, q, sigma, T):
    numerator = math.log(s_0/k) + ((r-q+((sigma**2)/2))*T)
    denominator = sigma*(T ** 0.5)
    return (numerator/denominator)

def d_2(s_0, k, r, q, sigma, T):
    numerator = math.log(s_0/k) + ((r-q-((sigma**2)/2))*T)
    denominator = sigma*(T ** 0.5)
    return (numerator/denominator)

def D_1(s_0, k, l, r, q, sigma, T):
    d_k = d_1(s_0, k, r, q, sigma, T)
    d_l = d_1(s_0, l, r, q, sigma, T)
    return (norm.cdf(d_k) - norm.cdf(d_l))

def D_2(s_0, k, l, r, q, sigma, T):
    d_k = d_2(s_0, k, r, q, sigma, T)
    d_l = d_2(s_0, l, r, q, sigma, T)
    return (norm.cdf(d_k) - norm.cdf(d_l))

# input
S_0 = 100
r = 0.05
q = 0.02
sigma = 0.5
T = 0.4
K_1 = 90
K_2 = 98
K_3 = 102
# K_4 = 110
K_4 = 104


# output
slope = (K_2-K_1) / (K_4-K_3)
D_2_12 = D_2(S_0, K_1, K_2, r, q, sigma, T)
D_2_23 = D_2(S_0, K_2, K_3, r, q, sigma, T)
D_2_34 = D_2(S_0, K_3, K_4, r, q, sigma, T)
price = (S_0*math.exp(0-(q*T))) * (D_1(S_0, K_1, K_2, r, q, sigma, T) - (slope*D_1(S_0, K_3, K_4, r, q, sigma, T)))
price = price + (math.exp(0-(r*T)) * (((K_2-K_1)*D_2_23) - (K_1*D_2_12) + (slope*K_4*D_2_34)))

print("The price of the derivative is:", end="\n")
print(price)