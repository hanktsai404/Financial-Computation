'''
Financial Computation: Extra Bonus 1
Author: Cheng-Han Tsai
b07703014@ntu.edu.tw
'''

from FINCOMP import Close_Form, Payoffs, BinTree

S_0 = 50
r = 0.1
q = 0.03
T = 0.5
K = 55
n_period = 100

eu_call = 2.5
am_put = 6.5

print("European Call Implied Volatility:")
eu_call_vol = Close_Form.implied_vol_bisection(S_0, K, r, q, T, eu_call, is_call=True) # One can use is_call=False if the option is a put
print("Black-Scholes (bisection method):", eu_call_vol)
eu_call_vol = Close_Form.implied_vol_Newton(S_0, K, r, q, T, eu_call, is_call=True) # One can use is_call=False if the option is a put
print("Black-Scholes (Newton's method):", eu_call_vol)
print()

implied_vol_calculator = BinTree.implied_vol_calculator(S_0, r, q, T, n_period)
eu_call_vol = implied_vol_calculator.implied_vol_bisection(eu_call, lambda x: Payoffs.eu_call_payoff(x, K), is_eu=True)
print("Binomial Tree (bisection method):", eu_call_vol)
eu_call_vol = implied_vol_calculator.implied_vol_Newton(eu_call, lambda x: Payoffs.eu_call_payoff(x, K), is_eu=True)
print("Binomial Tree (Newton's method):", eu_call_vol)
print()
print()

print("American Put Implied Volatility:")
am_put_vol = implied_vol_calculator.implied_vol_bisection(am_put, lambda x: Payoffs.eu_put_payoff(x, K), is_eu=False)
print("Binomial Tree (bisection method):", am_put_vol)
am_put_vol = implied_vol_calculator.implied_vol_Newton(am_put, lambda x: Payoffs.eu_put_payoff(x, K), is_eu=False)
print("Binomial Tree (Newton's method):", am_put_vol)


