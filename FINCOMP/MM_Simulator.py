'''
Monte-Carlo Simulator
'''

import numpy as np
from scipy.stats import norm
import math



class random_sample:
    def __init__(self):
        pass

    def anti_moment(self, N):
        '''Antithetic variate with moment matching approach'''
        if not N%2==0: size=N-1
        else: size = N
        sub_sample = np.random.normal(size=int(size/2))
        sample = np.append(sub_sample, (-1)*sub_sample)
        if not np.shape(sample)[0] == N: sample = np.append(sample, np.random.normal(size=1))
        m = np.mean(sample)
        s = np.std(sample)
        sample = (sample - m)/s
        return sample
    
    def cholesky(self, C, is_lower = True):
        '''Give lower-triangular matrix of Cholesky discomposition'''
        L = np.zeros(np.shape(C))
        n = np.shape(C)[0]
        L[0, 0] = np.sqrt(C[0, 0])
        # First row
        for j in range(1, n):
            L[0, j] = C[0, j]/L[0, 0]
        # From second row
        for i in range(1, n):
            for j in range(i, n):
                if i==j:
                    subtract = 0
                    for k in range(0, i): subtract = subtract + (L[k, i]**2)
                    L[i, j] = np.sqrt(C[i, j]-subtract)
                else:
                    subtract = 0
                    for k in range(0, i): subtract = subtract + (L[k, i]*L[k, j])
                    L[i, j] = (C[i, j] - subtract)/L[i, i]
        if is_lower: L = np.transpose(L) # To make it a lower-triangular matrix
        return L
    
    def inv_cholesky_sample(self, n_stocks, N):
        '''Inverse Cholesky Method by Wang (2008)'''
        sample = np.random.normal(size=(n_stocks, N))
        C_tilde = np.cov(sample)
        L_tilde = self.cholesky(C_tilde)
        inv_L_tilde = np.linalg.inv(L_tilde)
        sample = np.matmul(inv_L_tilde, sample)
        return sample


class stock_price_simulator:
    def __init__(self, s_0:float, mu:float, r: float, q: float, sigma: float):
        self.s_0 = s_0
        self.mu = mu
        self.r = r
        self.q = q
        self.sigma = sigma
    
    def simulate_price(self, T, N: int, ite: int, is_risk_neutral=True):
        price_sim = np.empty([N, ite])
        shift = self.mu
        if is_risk_neutral:
            shift = self.r
        mean = math.log(self.s_0) + ((shift-self.q-((self.sigma**2)/2))*T)
        sd = (self.sigma**2) * T
        sd = math.sqrt(sd)
        for i in range(0, ite):
            sample = norm.rvs(size=N, loc=mean, scale=sd)
            sim_price = list(map(math.exp, sample))
            price_sim[:,i] = np.array(sim_price)
        self.sim_stock_price = price_sim
    
    def deriv_pricing(self, T, payoff):
        discounted_payoffs = np.array([list(map(payoff, sim_price)) for sim_price in self.sim_stock_price])
        self.discounted_payoffs = np.array([list(map(lambda x: x*math.exp((0-self.r)*T), pf)) for pf in discounted_payoffs])
        deriv_prices = []
        for i in range(0, self.discounted_payoffs.shape[1]):
            deriv_prices.append(np.mean(self.discounted_payoffs[:,i]))
        self.deriv_prices = deriv_prices
    

class multi_stock_price_simulator:
    def __init__(self, s_0_arr, mu_arr, r, q_arr, sigma_arr, cov_matrix):
        self.s_0_arr = s_0_arr
        self.mu_arr = mu_arr
        self.r = r
        self.q_arr = q_arr
        self.sigma_arr = sigma_arr
        self.cov_matrix = cov_matrix
        self.rand = random_sample()
    
    def simulate_prices(self, T, N, ite, is_risk_neutral = True, is_anti_moment = False, is_inv_cholesky = False):
        shift = self.mu_arr
        if is_risk_neutral:
            shift = np.full(np.shape(self.s_0_arr), self.r)
        n_stocks = np.shape(self.s_0_arr)[0]
        means = np.log(self.s_0_arr)
        means = means + ((shift - self.q_arr - (0.5*np.square(self.sigma_arr)))*T)
        sim_stocks_prices = np.zeros([ite, n_stocks, N])
        L = self.rand.cholesky(self.cov_matrix)
        if is_inv_cholesky:
            for i in range(0, ite):
                sim_stocks_prices[i,:,:] = self.rand.inv_cholesky_sample(n_stocks, N)
        else:
            
            for i in range(0, ite):
                for s in range(0, n_stocks):
                    if is_anti_moment:
                        sim_stocks_prices[i,s,:] = self.rand.anti_moment(N)
                    else:
                        sim_stocks_prices[i,s,:] = np.random.normal(size=N)
            
        for i in range(0, ite): # iteration
            for j in range(0, N): # sample size
                # one column one sample
                sample = sim_stocks_prices[i,:,j]
                sample = np.matmul(L, sample)
                log_prices = sample + means
                sim_stocks_prices[i,:,j] = np.exp(log_prices)
        # print(sim_stocks_prices)
        self.sim_stocks_prices = sim_stocks_prices
    
    def deriv_pricing(self, T, payoff):
        ite = np.shape(self.sim_stocks_prices)[0]
        N = np.shape(self.sim_stocks_prices)[2]
        deriv_prices = np.zeros(ite)
        for i in range(0, ite):
            payoffs = np.zeros(N)
            for j in range(0, N):
                payoffs[j] = payoff(self.sim_stocks_prices[i,:,j])
            deriv_prices[i] = np.exp(-self.r*T)*np.mean(payoffs)
        return deriv_prices

class path_max_simulator:
    def __init__(self, s_t:float, mu:float, r: float, q: float, sigma: float) -> None:
        self.s_t = s_t
        self.mu = mu
        self.r = r
        self.q = q
        self.sigma = sigma
    
    def simulate_a_floating_lb_value(self, step_sim, T, t, n_period, s_max_t, is_risk_neutral=True):
        delta_t = (T-t)/n_period
        step_sim.s_0 = self.s_t
        max_price = s_max_t
        price = self.s_t
        for k in range(1, n_period+1):
            step_sim.simulate_price(delta_t, 1, 1, is_risk_neutral)
            price = np.asscalar(step_sim.sim_stock_price)
            step_sim.s_0 = price
            if price >= max_price: max_price = price
        return max_price - price
    
    def simulate_floating_lb_puts(self, T, t, s_max_t, n_period, N, ite, is_risk_neutral=True):
        self.floating_put_values = np.zeros([ite, N])
        shift = self.mu
        delta_t = (T-t)/n_period
        if is_risk_neutral: shift = self.r
        for i in range(0, ite):
            for j in range(0, N):
                this_price = self. s_t
                max_price = s_max_t
                sd = (self.sigma**2) * delta_t
                sd = math.sqrt(sd)
                for time in range(0, n_period+1):
                    mean = math.log(this_price) + ((shift-self.q-((self.sigma**2)/2))*delta_t)
                    sample = np.random.normal(loc=mean, scale=sd)
                    this_price = np.exp(sample)
                    if this_price >= max_price: max_price = this_price
                self.floating_put_values[i, j] = max_price - this_price
    
    def floating_lb_put_pricing(self, T, t):
        ite = np.shape(self.floating_put_values)[0]
        self.deriv_prices = np.zeros(ite)
        for i in range(0, ite):
            this_deeriv_prices = self.floating_put_values[i,:]
            self.deriv_prices[i] = np.mean(this_deeriv_prices) * np.exp(0-(self.r*(T-t)))

class path_arith_ave_simulator:
    def __init__(self, s_t:float, mu:float, r: float, q: float, sigma: float):
        self.s_t = s_t
        self.mu = mu
        self.r = r
        self.q = q
        self.sigma = sigma
    
    def simulate_ave_deriv_values(self, T, t, s_ave_t, n_period, N, ite, payoff, is_risk_neutral=True):
        self.ave_arith_deriv_values = np.zeros([ite, N])
        delta_t = (T-t)/n_period
        total_step = round(T/delta_t)
        shift = self.mu
        if is_risk_neutral: shift = self.r
        sd = (self.sigma**2) * delta_t
        sd = math.sqrt(sd)
        for i in range(0, ite):
            for j in range(0, N):
                this_price = self.s_t
                price_sum = s_ave_t*(total_step - n_period + 1)
                for k in range(0, n_period):
                    mean = math.log(this_price) + ((shift-self.q-((self.sigma**2)/2))*delta_t)
                    sample = np.random.normal(loc=mean, scale=sd)
                    this_price = np.exp(sample)
                    price_sum = price_sum + this_price
                ave_price = price_sum/(total_step+1)
                self.ave_arith_deriv_values[i, j] = payoff(ave_price)
    
    def ave_deriv_pricing(self, T, t):
        ite = np.shape(self.ave_arith_deriv_values)[0]
        self.deriv_prices = np.zeros(ite)
        for i in range(0, ite):
            self.deriv_prices[i] = np.mean(self.ave_arith_deriv_values[i,:])




