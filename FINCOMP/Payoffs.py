'''
Payoffs function of different options
'''

def eu_call_payoff(x, k):
    if x>=k:
        return (x-k)
    else:
        return 0

def eu_put_payoff(x, k):
    if x<=k:
        return (k-x)
    else:
        return 0

def eu_rainbow_call_payoff(under_prices, k, onmax=True):
    if onmax:
        x = max(under_prices)
    else:
        x = min(under_prices)
    return eu_call_payoff(x, k)

def eu_rainbow_put_payoff(under_prices, k, onmax=True):
    if onmax:
        x = max(under_prices)
    else:
        x = min(under_prices)
    return eu_put_payoff(x, k)

def floating_lb_call_payoff(s_min, x):
    return x-s_min

def floating_lb_put_payoff(s_max, x):
    return s_max-x

def fixed_lb_call_payoff(s_max, k):
    if s_max-k>0:
        return s_max-k
    else:
        return 0
