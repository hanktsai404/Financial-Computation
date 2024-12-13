'''
Binomial Trees
'''

import numpy as np
from math import exp, sqrt
from scipy.stats import binom

def lin_inter(x, x_0, y_0, x_1, y_1):
    if x_1-x_0==0: return y_0
    else: return y_0 + ((x-x_0)*(y_1-y_0)/(x_1-x_0))

def lin_inter_search(data, key): # data is from max to min, equally populated
    max_value = data[0]
    min_value = data[len(data)-1]
    gap = (max_value - min_value) / (len(data) - 1)
    if gap==0: idx = 0
    else:
        idx = (max_value - key)/gap
    
    if int(idx)==len(data)-1: idx=idx-1
    return int(idx)

def binary_search(data, key):
    if data[len(data)-1]-data[0]==0: return 0
    temp_data = data
    start_idx = 0
    idx = 0
    while True:
        if len(temp_data)==2:
            idx=start_idx
            break
        mid_idx = int(len(temp_data)/2)
        if temp_data[mid_idx] < key:
            temp_data = np.copy(temp_data[:mid_idx+1])
        else:
            temp_data = np.copy(temp_data[mid_idx:])
            start_idx = start_idx + mid_idx
    return idx

def sequential_search(data, key):
    if data[len(data)-1]-data[0]==0 or key>=data[0]: return 0
    for i in range(0, len(data)-1):
        if data[i] == key: return i
        elif data[i]>key and data[i+1]<=key: return i
        
        if i==len(data)-2: return i

def derivative(x, func):
    delta_x = 10**(-10)
    return (func(x+delta_x) - func(x)) / delta_x

class CRR_tree:
    def __init__(self, s_0:float, r:float, q:float, sigma:float, T:float, n_period:int):
        '''Notice: Trees grow downward! (This is to comply with practices of the other algorithms)'''
        self.delta_t = T/n_period
        self.u = exp(sigma*sqrt(self.delta_t))
        self.d = 1/self.u
        self.prob = (exp((r-q)*self.delta_t)-self.d) / (self.u-self.d)
        self.n_period = n_period
        self.s_0 = s_0
        self.r = r
        self.asset_tree = np.zeros([self.n_period+1, self.n_period+1])
    
    def grow_simple_tree(self):
        for i in range(0, self.n_period+1):
            for j in range(0, i+1):
                self.asset_tree[i, j] = self.s_0 * (self.u**(i-j)) * (self.d**j)

class backward_inductor_2d:
    def __init__(self, tree:CRR_tree):
        self.asset_tree = tree
    
    def backward(self, payoff, is_eu = True):
        '''Payoff should include the strike price, as a function of S_T'''
        n_period = np.shape(self.asset_tree.asset_tree)[0] - 1
        self.deriv_price_tree = np.copy(self.asset_tree.asset_tree)
        # Backward induction
        self.deriv_price_tree[n_period, :] = list(map(payoff, self.deriv_price_tree[n_period, :]))
        for i in range(n_period-1, -1, -1): # the loop works backward
            for j in range(0, i+1):
                temp = (self.asset_tree.prob * self.deriv_price_tree[i+1, j]) + ((1-self.asset_tree.prob) * self.deriv_price_tree[i+1, j+1])
                self.deriv_price_tree[i, j] = exp((0-self.asset_tree.r)*self.asset_tree.delta_t)*temp
                if not is_eu: # For American
                    if self.deriv_price_tree[i, j] <= payoff(self.asset_tree.asset_tree[i, j]):
                        self.deriv_price_tree[i, j] = payoff(self.asset_tree.asset_tree[i, j])
        return self.deriv_price_tree[0, 0]

class backward_inductor_1d:
    def __init__(self, tree:CRR_tree):
        self.asset_tree = tree
    
    def backward(self, payoff, is_eu = True):
        n_period = np.shape(self.asset_tree.asset_tree)[0] - 1
        self.deriv_price_vector = np.copy(self.asset_tree.asset_tree[n_period, :])
        self.deriv_price_vector = np.array(list(map(payoff, self.deriv_price_vector)))
        for i in range(n_period-1, -1, -1):
            for j in range(0, i+1):
                temp = (self.asset_tree.prob * self.deriv_price_vector[j]) + ((1-self.asset_tree.prob) * self.deriv_price_vector[j+1])
                self.deriv_price_vector[j] = exp((0-self.asset_tree.r)*self.asset_tree.delta_t)*temp
                if not is_eu:
                    if self.deriv_price_vector[j] <= payoff(self.asset_tree.asset_tree[i, j]):
                        self.deriv_price_vector[j] = payoff(self.asset_tree.asset_tree[i, j])
        return self.deriv_price_vector[0]

class implied_vol_calculator:
    def __init__(self, s_0:float, r:float, q:float, T:float, n_period:int):
        self.s_0 = s_0
        self.r = r
        self.q = q
        self.T = T
        self.n_period = n_period
    
    def _price(self, sigma, payoff, is_eu=True):
        asset_tree = CRR_tree(self.s_0, self.r, self.q, sigma, self.T, self.n_period)
        asset_tree.grow_simple_tree()
        inductor = backward_inductor_1d(asset_tree)
        return inductor.backward(payoff, is_eu)
    
    def implied_vol_bisection(self, option_price, payoff, is_eu=True):
        price_func = lambda x: self._price(x, payoff, is_eu) - option_price
        upper = 1
        lower = 10**(-4)
        while True:
            mid = (upper+lower) / 2
            if price_func(lower)*price_func(mid) < 0: upper = mid
            else: lower = mid
            if upper-lower < 10**(-8): break
        return (upper+lower) / 2
    
    def implied_vol_Newton(self, option_price, payoff, is_eu=True):
        price_func = lambda x: self._price(x, payoff, is_eu) - option_price
        deriv = lambda y: derivative(y, price_func)
        sigma = 0.8
        last_sigma = sigma
        while True:
            sigma = sigma - (price_func(sigma)/deriv(sigma))
            if last_sigma-sigma < 10**(-11): break
            last_sigma = sigma
        return sigma



def eu_comb_method(s_0:float, r:float, q:float, sigma:float, T:float, n_period:int, payoff):
    delta_t = T/n_period
    u = exp(sigma*sqrt(delta_t))
    d = 1/u
    prob = (exp((r-q)*delta_t)-d) / (u-d)
    price = 0
    for j in range(0, n_period+1):
        price = price + binom.pmf(j, n_period, 1-prob) * payoff(s_0*(u ** (n_period-j))*(d ** j))
    price = price * exp((0-r)*T)
    return price


class path_dependent_CRR_tree:
    def __init__(self, s_t:float, r:float, q:float, sigma:float, T:float, n_period:int, t:float, s_pathdep_t:float):
        '''Notice: Trees grow downward! (This is to comply with practices of the other algorithms)'''
        self.delta_t = (T-t)/n_period
        self.total_step = round(T/self.delta_t)
        self.u = exp(sigma*sqrt(self.delta_t))
        self.d = 1/self.u
        self.prob = (exp((r-q)*self.delta_t)-self.d) / (self.u-self.d)
        self.n_period = n_period
        self.s_t = s_t
        self.r = r
        self.s_maxmin_t = s_pathdep_t
    
    def grow_maxmin_tree(self, is_max):
        self.asset_tree = np.zeros([self.n_period+1, self.n_period+1, 2*self.n_period+3])
        self.asset_tree[0, 0, 0] = self.s_t
        self.asset_tree[0, 0, 1] = self.s_maxmin_t
        maxmin_len = np.shape(self.asset_tree)[2]
        for i in range(1, self.n_period+1):
            for j in range(0, i+1):
                if j==0:
                    self.asset_tree[i, j, 0] = self.asset_tree[i-1, j, 0] * self.u
                    if is_max:
                        self.asset_tree[i, j, 1] = max(self.asset_tree[i, j, 0], self.s_maxmin_t) # Highest ever
                    else:
                        self.asset_tree[i, j, 1] = self.asset_tree[0, 0, 1]
                elif j==i:
                    self.asset_tree[i, j, 0] = self.asset_tree[i-1, j-1, 0] * self.d
                    if is_max:
                        self.asset_tree[i, j, 1] = self.asset_tree[0, 0, 1]
                    else:
                        self.asset_tree[i, j, 1] = min(self.asset_tree[i, j, 0], self.s_maxmin_t)
                else:
                    self.asset_tree[i, j, 0] = self.asset_tree[i-2, j-1, 0]
                    # Sort the max values
                    if is_max:
                        l = 1
                        m = 1
                        for k in range(1, maxmin_len):
                            if k-1!=0 and self.asset_tree[i, j, k-1]==self.asset_tree[i, j, 0]: break
                            if k-1!=0 and self.asset_tree[i-1, j-1, l]==self.asset_tree[i, j, k-1]: l=l+1
                            if k-1!=0 and self.asset_tree[i-1, j, m]==self.asset_tree[i, j, k-1]: m=m+1

                            if self.asset_tree[i-1, j-1, l]>=self.asset_tree[i-1, j, m] and self.asset_tree[i-1, j-1, l]>self.asset_tree[i, j, 0]:
                                self.asset_tree[i, j, k] = self.asset_tree[i-1, j-1, l]
                                l = l+1
                            elif self.asset_tree[i-1, j, m]>=self.asset_tree[i-1, j-1, l] and self.asset_tree[i-1, j, m]>self.asset_tree[i, j, 0]:
                                self.asset_tree[i, j, k] = self.asset_tree[i-1, j, m]
                                m = m+1
                            else:
                                if self.asset_tree[i-1, j-1, l]==0 and self.asset_tree[i-1, j, m]==0:
                                    break
                                else:
                                    self.asset_tree[i, j, k] = self.asset_tree[i, j, 0]
                                    break
                    else:
                        l = 1
                        m = 1
                        for k in range(1, maxmin_len):
                            if k-1!=0 and self.asset_tree[i-1, j-1, l]==self.asset_tree[i, j, k-1]: l=l+1
                            if k-1!=0 and self.asset_tree[i-1, j, m]==self.asset_tree[i, j, k-1]: m=m+1

                            if self.asset_tree[i-1, j-1, l]<=self.asset_tree[i-1, j, m] and self.asset_tree[i-1, j-1, l]<self.asset_tree[i, j, 0]:
                                self.asset_tree[i, j, k] = self.asset_tree[i-1, j-1, l]
                                l = l+1
                            elif self.asset_tree[i-1, j, m]<=self.asset_tree[i-1, j-1, l] and self.asset_tree[i-1, j, m]<self.asset_tree[i, j, 0]:
                                self.asset_tree[i, j, k] = self.asset_tree[i-1, j, m]
                                m = m+1
                            else:
                                if self.asset_tree[i-1, j-1, l]==0 and self.asset_tree[i-1, j, m]==0:
                                    break
                                else:
                                    self.asset_tree[i, j, k] = self.asset_tree[i, j, 0]
                                    break

    def grow_maxmin_tree_fast(self, is_max):
        self.asset_tree = np.zeros([self.n_period+1, self.n_period+1, 2*self.n_period+3])
        self.asset_tree[0, 0, 0] = self.s_t
        self.asset_tree[0, 0, 1] = self.s_maxmin_t
        for i in range(1, self.n_period+1):
            for j in range(0, i+1):
                if j==0:
                    self.asset_tree[i, j, 0] = self.asset_tree[i-1, j, 0] * self.u
                elif j==i:
                    self.asset_tree[i, j, 0] = self.asset_tree[i-1, j-1, 0] * self.d
                else:
                    self.asset_tree[i, j, 0] = self.asset_tree[i-2, j-1, 0]
                
                # dertermine S_max for each node
                if is_max:
                    for l in range(0, j+1):
                        if self.asset_tree[i-(j-l), l, 0] <= self.asset_tree[0, 0, 1]: break
                        self.asset_tree[i, j, l+1] = self.asset_tree[i-(j-l), l, 0]
                # determine S_min for each node
                else:
                    for t in range(j, i+1):
                        if self.asset_tree[t, j, 0] >= self.asset_tree[0, 0, 1]: break
                        self.asset_tree[i, j, t-j+1] = self.asset_tree[t, j, 0]

    def floating_lb_backward_induction(self, is_eu, is_call):
        maxmin_len = np.shape(self.asset_tree)[2]
        self.option_prices = np.zeros([self.n_period+1, maxmin_len]) # We don't need the time dimension
        # At expiry
        for j in range(0, self.n_period+1):
            for k in range(0, maxmin_len):
                if self.asset_tree[self.n_period, j, k]==0:break
                if not is_call:
                    if k==0: self.option_prices[j, k] = self.asset_tree[self.n_period, j, k]
                    else: self.option_prices[j, k] = self.asset_tree[self.n_period, j, k] - self.asset_tree[self.n_period, j, 0]
                else:
                    if k==0: self.option_prices[j, k] = self.asset_tree[self.n_period, j, k]
                    else: self.option_prices[j, k] = self.asset_tree[self.n_period, j, 0] - self.asset_tree[self.n_period, j, k]

        # Not at expiry (backward induction)
        for i in range(self.n_period-1, -1, -1):
            for j in range(0, i+1):
                this_option_price = np.zeros(maxmin_len)
                this_option_price[0] = self.asset_tree[i, j, 0]
                for k in range(1, maxmin_len):
                    if self.asset_tree[i, j, k]==0: break
                    if not is_call:
                        upper_put = 0
                        lower_put = 0

                        if self.asset_tree[i, j, k] not in self.asset_tree[i+1, j, 1:]:
                            is_find_lower = False
                            is_find_upper = False
                            for l in range(1, maxmin_len):
                                if self.asset_tree[i+1, j, l] == self.asset_tree[i+1, j, 0]:
                                    upper_put = self.option_prices[j, l]
                                    is_find_upper = True
                                if self.asset_tree[i+1, j+1, l] == self.asset_tree[i, j, k]:
                                    lower_put = self.option_prices[j+1, l]
                                    is_find_lower = True
                                if is_find_lower and is_find_upper: break
                        else:
                            is_find_lower = False
                            is_find_upper = False
                            for l in range(1, maxmin_len):
                                if self.asset_tree[i+1, j, l] == self.asset_tree[i, j, k]:
                                    upper_put = self.option_prices[j, l]
                                    is_find_upper = True
                                if self.asset_tree[i+1, j+1, l] == self.asset_tree[i, j, k]:
                                    lower_put = self.option_prices[j+1, l]
                                    is_find_lower = True
                                if is_find_lower and is_find_upper: break
                        
                        
                        temp = (self.prob*upper_put) + ((1-self.prob)*lower_put)
                        this_option_price[k] = exp((0-self.r)*self.delta_t) * temp

                        if not is_eu:
                            this_option_price[k] = max(this_option_price[k], (self.asset_tree[i, j, k]-self.asset_tree[i, j, 0]))
                    
                    else:
                        is_find_upper = False
                        is_find_lower = False
                        upper_call = 0
                        lower_call = 0

                        if self.asset_tree[i, j, k] not in self.asset_tree[i+1, j+1, 1:]:
                            is_find_lower = False
                            is_find_upper = False
                            for l in range(1, maxmin_len):
                                if self.asset_tree[i+1, j+1, l] == self.asset_tree[i+1, j+1, 0]:
                                    lower_call = self.option_prices[j+1, l]
                                    is_find_lower = True
                                if self.asset_tree[i+1, j, l] == self.asset_tree[i, j, k]:
                                    upper_call = self.option_prices[j, l]
                                    is_find_upper = True
                                if is_find_lower and is_find_upper: break
                        else:
                            is_find_lower = False
                            is_find_upper = False
                            for l in range(1, maxmin_len):
                                if self.asset_tree[i+1, j+1, l] == self.asset_tree[i, j, k]:
                                    lower_call = self.option_prices[j+1, l]
                                    is_find_lower = True
                                if self.asset_tree[i+1, j, l] == self.asset_tree[i, j, k]:
                                    upper_call = self.option_prices[j, l]
                                    is_find_upper = True
                                if is_find_lower and is_find_upper: break
                        
                        temp = (self.prob*upper_call) + ((1-self.prob)*lower_call)
                        this_option_price[k] = np.exp((0-self.r)*self.delta_t) * temp
                        if is_find_upper and is_find_lower: break

                        if not is_eu:
                            this_option_price[k] = max(this_option_price[k], self.asset_tree[i, j, 0]-self.asset_tree[i, j, k])
                
                self.option_prices[j,:] = np.copy(this_option_price)
        
        return self.option_prices[0, 1]

    def grow_arith_ave_tree(self, M, is_linear=True):
        self.is_linear = is_linear
        s_ave_t = self.s_maxmin_t
        self.asset_tree = np.zeros([self.n_period+1, self.n_period+1, M+2])
        self.asset_tree[0, 0, 0] = self.s_t
        self.asset_tree[0, 0, 1] = s_ave_t
        for i in range(1, self.n_period+1):
            for j in range(0, i+1):
                if j==0:
                    self.asset_tree[i, j, 0] = self.asset_tree[i-1, j, 0] * self.u
                elif j==i:
                    self.asset_tree[i, j, 0] = self.asset_tree[i-1, j-1, 0] * self.d
                else:
                    self.asset_tree[i, j, 0] = self.asset_tree[i-2, j-1, 0]
                
                # Determine max and min of average
                ave_max = self.s_t + (self.s_t*self.u*(1-(self.u**(i-j)))/(1-self.u))
                ave_max = ave_max + (self.s_t*(self.u**(i-j))*self.d*(1-(self.d**j))/(1-self.d))
                ave_max = (ave_max - self.s_t + (s_ave_t*(self.total_step-self.n_period+1))) / (self.total_step-self.n_period+i+1)

                ave_min = self.s_t + (self.s_t*self.d*(1-(self.d**j))/(1-self.d))
                ave_min = ave_min + (self.s_t*(self.d**j)*self.u*(1-(self.u**(i-j)))/(1-self.u))
                ave_min = (ave_min - self.s_t + (s_ave_t*(self.total_step-self.n_period+1))) / (self.total_step-self.n_period+i+1)

                self.asset_tree[i, j, 1] = ave_max
                self.asset_tree[i, j, M+1] = ave_min
                if is_linear:
                    gap = (ave_max-ave_min)/M
                    ave = ave_max - gap
                    for k in range(2, M+1):
                        self.asset_tree[i, j, k] = ave
                        ave = ave - gap
                else:
                    for k in range(2, M+1):
                        temp = ((M-k+1)*np.log(ave_max)/M) + ((k-1)*np.log(ave_min)/M)
                        self.asset_tree[i, j, k] = np.exp(temp)
    
    def asian_backward_induction(self, is_eu, payoff, is_seq_search=False, is_bin_search=False):
        '''If is_seq_search and is_bin_search all both False, linear interpolation is applied'''
        M = np.shape(self.asset_tree)[2] - 2
        
        self.option_prices = np.zeros([self.n_period+1, M+2])
        # At expiry
        for j in range(0, self.n_period+1):
            self.option_prices[j, 0] = self.asset_tree[self.n_period, j, 0]
            for k in range(1, M+2):
                self.option_prices[j, k] = payoff(self.asset_tree[self.n_period, j, k])
        
        # Not at expiry
        for i in range(self.n_period-1, -1, -1):
            for j in range(0, i+1):
                this_option_prices = np.zeros(M+2)
                this_option_prices[0] = self.asset_tree[i, j, 0]
                for k in range(1, M+2):
                    if self.asset_tree[i, j, k] == 0: break
                    ave_upper = (self.asset_tree[i, j, k]*(self.total_step-self.n_period+i+1)) + self.asset_tree[i+1, j, 0]
                    ave_upper = ave_upper / (self.total_step-self.n_period+i+2)
                    if (not is_seq_search) and (not is_bin_search):
                        if self.is_linear:
                            idx = lin_inter_search(self.asset_tree[i+1, j, 1:], ave_upper) + 1
                        else:
                            idx = lin_inter_search(np.log(np.copy(self.asset_tree[i+1, j, 1:])), np.log(ave_upper)) + 1
                    elif is_bin_search:
                        idx = binary_search(self.asset_tree[i+1, j, 1:], ave_upper) + 1
                    else:
                        idx = sequential_search(self.asset_tree[i+1, j, 1:], ave_upper) + 1
                    

                    if self.asset_tree[i+1, j, 1]==self.asset_tree[i+1, j, M+1]:
                        upper_option_value = self.option_prices[j, 1]
                    else:
                        upper_option_value = lin_inter(ave_upper, self.asset_tree[i+1, j, idx+1], self.option_prices[j, idx+1], self.asset_tree[i+1, j, idx], self.option_prices[j, idx])

                    ave_lower = (self.asset_tree[i, j, k]*(self.total_step-self.n_period+i+1)) + self.asset_tree[i+1, j+1, 0]
                    ave_lower = ave_lower / (self.total_step-self.n_period+i+2)
                    if (not is_seq_search) and (not is_bin_search):
                        if self.is_linear:
                            idx = lin_inter_search(self.asset_tree[i+1, j+1, 1:], ave_lower) + 1
                        else:
                            idx = lin_inter_search(np.log(np.copy(self.asset_tree[i+1, j+1, 1:])), np.log(ave_lower)) + 1
                    elif is_bin_search:
                        idx = binary_search(self.asset_tree[i+1, j+1, 1:], ave_lower) + 1
                    else:
                        idx = sequential_search(self.asset_tree[i+1, j+1, 1:], ave_lower) + 1
                        
                    
                    if self.asset_tree[i+1, j, 1]==self.asset_tree[i+1, j, M+1]:
                        lower_option_value = self.option_prices[j+1, 1]
                    else:
                        lower_option_value = lin_inter(ave_lower, self.asset_tree[i+1, j+1, idx+1], self.option_prices[j+1, idx+1], self.asset_tree[i+1, j+1, idx], self.option_prices[j+1, idx])

                    
                    temp = (self.prob*upper_option_value) + ((1-self.prob)*lower_option_value)
                    this_option_prices[k] = np.exp(0-(self.r*self.delta_t)) * temp

                    if not is_eu:
                        this_option_prices[k] = max(this_option_prices[k], payoff(self.asset_tree[i, j, k]))
                self.option_prices[j,:] = np.copy(this_option_prices)
        
        return self.option_prices[0, 1]


def CV_floating_lb_put(s_t:float, r:float, q:float, sigma:float, T:float, n_period:int, t:float, is_eu:bool):
    '''Cheuk and Vorst (1997) method to price European and American floating lookback put. The tree grows rightward to comply with the paper. The method
    does not allow historical max and current spot price at t to be different'''
    delta_t = (T-t) / n_period
    u = exp(sigma*sqrt(delta_t))
    d = 1/u
    mu = exp((r-q)*delta_t)
    r = exp(r*delta_t)
    prob_u = (mu*u - 1)/(mu*(u-d))

    # No need to grow a tree
    # Backward induction
    deriv_prices = np.zeros([n_period+1, 2]) # The first column is the target, the second column is just a temperary space from the last step.
    for i in range(0, n_period+1):
        deriv_prices[i, 1] = max((u**i) - 1, 0)
    
    for time_idx in range(n_period-1, -1, -1):
        for i in range(0, time_idx+1):
            if i == 0:
                temp = (prob_u*deriv_prices[i, 1]) + ((1-prob_u)*deriv_prices[i+1, 1])
            else:
                temp = (prob_u*deriv_prices[i-1, 1]) + ((1-prob_u)*deriv_prices[i+1, 1])
            deriv_prices[i, 0] = temp * mu/r

            if not is_eu:
                deriv_prices[i, 0] = max(deriv_prices[i, 0], u**i-1)
        deriv_prices[:,1] = deriv_prices[:,0]
    return s_t*deriv_prices[0, 0]
            





if __name__ == "__main__":
    data = [10, 8, 6, 4, 2, 0]
    print(sequential_search(data, 5))

    


