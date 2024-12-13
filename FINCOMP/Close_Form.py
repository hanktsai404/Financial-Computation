'''
Close Form Price for Options
'''

import math
from scipy.stats import norm
import numpy as np


def _d1(s_0, k, r, q, sigma, T):
    numerator = math.log(s_0/k) + ((r-q+((sigma**2)/2))*T)
    denominator = sigma*(T ** 0.5)
    return (numerator/denominator)

def _d2(s_0, k, r, q, sigma, T):
    numerator = math.log(s_0/k) + ((r-q-((sigma**2)/2))*T)
    denominator = sigma*(T ** 0.5)
    return (numerator/denominator)

def eu_call_price(s_0, k, r, q, sigma, T):
    d_1 = _d1(s_0, k, r, q, sigma, T)
    d_2 = _d2(s_0, k, r, q, sigma, T)
    call_price = s_0*(math.exp((0-q)*T))*norm.cdf(d_1)
    call_price = call_price - (k*math.exp((0-r)*T)*norm.cdf(d_2))
    return call_price

def eu_put_price(s_0, k, r, q, sigma, T):
    call_price = eu_call_price(s_0, k, r, q, sigma, T)
    put_price = call_price + (k*math.exp((0-r)*T)) - (s_0*math.exp((0-q)*T))
    return(put_price)

def vega(s_0, k, r, q, sigma, T):
    d_1 = _d1(s_0, k, r, q, sigma, T)
    result = math.exp(0-(q*T)) * s_0 * norm.pdf(d_1)
    return result

def implied_vol_bisection(s_0, k, r, q, T, option_price, is_call=True):
    if is_call: price_func = lambda x: eu_call_price(s_0, k, r, q, x, T) - option_price
    else: price_func = lambda x: eu_put_price(s_0, k, r, q, x, T) - option_price
    upper = 1
    lower = 10**(-4)
    while True:
        mid = (upper+lower) / 2
        if price_func(lower)*price_func(mid) < 0: upper = mid
        else: lower = mid
        if upper-lower < 10**(-8): break
    return (upper+lower) / 2

def implied_vol_Newton(s_0, k, r, q, T, option_price, is_call=True):
    if is_call: price_func = lambda x: eu_call_price(s_0, k, r, q, x, T) - option_price
    else: price_func = lambda x: eu_put_price(s_0, k, r, q, x, T) - option_price
    sigma = 1
    last_sigma = sigma
    while True:
        sigma = sigma - (price_func(sigma)/vega(s_0, k, r, q, sigma, T))
        if last_sigma-sigma < 10**(-8): break
        last_sigma = sigma
    return sigma

if __name__=="__main__":
    print(implied_vol_bisection(300, 250, 0.03, 0.01, 1, 56.05))
    print()
    print(_d2(38, 40, 0.16, 0, 0.35, 0.5), end = "\n")
    print(norm.cdf(_d2(38, 40, 0.16, 0, 0.35, 0.5)), end = "\n")