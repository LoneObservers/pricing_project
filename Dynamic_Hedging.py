#!/usr/bin/env python
# coding: utf-8
### pricing project 2
### Oct 20, 2021
import pandas as pd
import numpy as np
import datetime as dt
#from scipy import linalg
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm  #### loop 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def Greeks_func(S, K, r, T, sigma, otype='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma*sigma)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if otype == 'call':
        oprice = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
                             
    elif otype == 'put':
        oprice = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
        
    else:
        return 'wrong type!'
    
    return oprice,delta,gamma


def time_based_h(mu, K, r, T, T_s, sigma, sigma_q, S0, scost, ocost, Nsteps=100):
    t = np.linspace(0,T,Nsteps+1)
    dt = t[1] - t[0]
    df = np.zeros((13,len(t))) 
    ### index: S, put_delta, put_gamma, call_delta, call_gamma, put_price,call_price, alpha_D, beta_D,
    ### alpha_D-G, beta_D-G, B1, B2

    df[0,0] = S0 # asset price at the beginning
    put_price, p_delta, p_gamma = Greeks_func(df[0,0], K, r, T, sigma_q, otype='put')
    call_price, c_delta, c_gamma = Greeks_func(df[0,0], K, r, T_s, sigma_q, otype='call')
    df[1,0] = p_delta ### put_delta
    df[2,0] = p_gamma ### put_gamma
    df[3,0] = c_delta ### call_delta
    df[4,0] = c_gamma ### call_gamma
    df[5,0] = put_price 
    df[6,0] = call_price
    df[7,0] = p_delta ### alpha of delta hedging
    df[8,0] = 0 ### beta of delta hedging
    
    df[11,0] = df[5,0] - df[7,0]*df[0,0] - abs(df[7,0])*scost
    df[10,0] = df[2,0]/df[4,0] ### beta of delta-gamma hedging
    df[9,0] = df[1,0] - df[10,0]*df[3,0] ### alpha of delta-gamma hedging
    df[12,0] = df[5,0] - df[9,0]*df[0,0] - df[10,0]*df[6,0] - abs(df[9,0])*scost - abs(df[10,0])*ocost
    
    ### dynamic hedging
    for j in range(Nsteps-1):
        df[0,j+1] = df[0,j]*np.exp((mu-0.5*sigma*sigma)*dt + sigma*np.sqrt(dt)*np.random.randn()) 
        put_price, p_delta, p_gamma = Greeks_func(df[0,j+1], K, r, T-(j+1)*dt, sigma_q, otype='put')
        call_price, c_delta, c_gamma = Greeks_func(df[0,j+1], K, r, T_s-(j+1)*dt, sigma_q, otype='call')

        df[1,j+1] = p_delta 
        df[2,j+1] = p_gamma 
        df[3,j+1] = c_delta 
        df[4,j+1] = c_gamma 
        df[5,j+1] = put_price 
        df[6,j+1] = call_price
        df[7,j+1] = p_delta 
        df[8,j+1] = 0 

        df[11,j+1] = df[11,j]*np.exp(r*dt) - (df[7,j+1] - df[7,j])*df[0,j+1] - abs(df[7,j+1] - df[7,j])*scost
        df[10,j+1] = df[2,j+1]/df[4,j+1]
        df[9,j+1] = df[1,j+1] - df[10,j+1]*df[3,j+1]
        df[12,j+1] = df[12,j]*np.exp(r*dt) - (df[9,j+1] - df[9,j])*df[0,j+1] - (df[10,j+1] - df[10,j])*df[6,j+1] - abs(df[9,j+1] - df[9,j])*scost - abs(df[10,j+1] - df[10,j])*ocost
        
    ### calculating pnl
    df[0,-1] = df[0,-2]*np.exp((mu-0.5*sigma*sigma)*dt + sigma*np.sqrt(dt)*np.random.randn()) 
    call_price, c_delta, c_gamma = Greeks_func(df[0,-1], K, r, T_s-T, sigma_q, otype='call')
    df[1,-1] = np.nan 
    df[2,-1] = np.nan  
    df[3,-1] = np.nan  
    df[4,-1] = np.nan  
    df[5,-1] = np.nan  
    df[6,-1] = call_price
    df[7,-1] = np.nan  
    df[8,-1] = np.nan  

    df[11,-1] = df[11,-2]*np.exp(r*dt) + df[7,-2]*df[0,-1] - np.maximum(K-df[0,-1],0) - abs(df[7,-2])*scost 
    df[10,-1] = np.nan
    df[9,-1] = np.nan
    df[12,-1] = df[12,-2]*np.exp(r*dt) + df[9,-2]*df[0,-1] + df[10,-2]*df[6,-1] - p.maximum(K-df[0,-1],0) - abs(df[9,-2])*scost - abs(df[10,-2])*ocost    
    return df




def move_based_h(mu, K, r, T, T_s, sigma, sigma_q, S0, scost, ocost, band, Nsteps=100):
    t = np.linspace(0,T,Nsteps+1)
    dt = t[1] - t[0]
    df = np.zeros((13,len(t))) 
    ### index: S, put_delta, put_gamma, call_delta, call_gamma, put_price,call_price, alpha_D, beta_D,
    ### alpha_D-G, beta_D-G, B1, B2

    df[0,0] = S0 # asset price at the beginning
    put_price, p_delta, p_gamma = Greeks_func(df[0,0], K, r, T, sigma_q, otype='put')
    call_price, c_delta, c_gamma = Greeks_func(df[0,0], K, r, T_s, sigma_q, otype='call')
    df[1,0] = p_delta ### put_delta
    df[2,0] = p_gamma ### put_gamma
    df[3,0] = c_delta ### call_delta
    df[4,0] = c_gamma ### call_gamma
    df[5,0] = put_price 
    df[6,0] = call_price
    df[7,0] = p_delta ### alpha of delta hedging
    df[8,0] = 0 ### beta of delta hedging
    
    df[11,0] = df[5,0] - df[7,0]*df[0,0] - abs(df[7,0])*scost
    df[10,0] = df[2,0]/df[4,0] ### beta of delta-gamma hedging
    df[9,0] = df[1,0] - df[10,0]*df[3,0] ### alpha of delta-gamma hedging
    df[12,0] = df[5,0] - df[9,0]*df[0,0] - df[10,0]*df[6,0] - abs(df[9,0])*scost - abs(df[10,0])*ocost
    
    upper_bound_D = df[7,0] + band/2
    lower_bound_D = df[7,0] - band/2
    upper_bound_DG = df[9,0] + band/2
    lower_bound_DG = df[9,0] - band/2
    
    ### dynamic hedging
    for j in range(Nsteps-1):
        df[0,j+1] = df[0,j]*np.exp((mu-0.5*sigma*sigma)*dt + sigma*np.sqrt(dt)*np.random.randn()) 
        put_price, p_delta, p_gamma = Greeks_func(df[0,j+1], K, r, T-(j+1)*dt, sigma_q, otype='put')
        call_price, c_delta, c_gamma = Greeks_func(df[0,j+1], K, r, T_s-(j+1)*dt, sigma_q, otype='call')

        df[1,j+1] = p_delta 
        df[2,j+1] = p_gamma 
        df[3,j+1] = c_delta 
        df[4,j+1] = c_gamma 
        df[5,j+1] = put_price 
        df[6,j+1] = call_price
        
        if (p_delta <= upper_bound_D) and (p_delta >= lower_bound_D):
            df[7,j+1] = df[7,j] 
            df[8,j+1] = df[8,j]  
        else:
            df[7,j+1] = p_delta 
            df[8,j+1] = 0 
            upper_bound_D = df[7,j+1] + band/2
            lower_bound_D = df[7,j+1] - band/2
            
        if (df[1,j+1] - df[2,j+1]*df[3,j+1]/df[4,j+1] <= upper_bound_DG) and (df[1,j+1] - df[2,j+1]*df[3,j+1]/df[4,j+1] >= lower_bound_DG):
            df[10,j+1] = df[10,j]
            df[9,j+1] = df[9,j]
        else:
            df[10,j+1] = df[2,j+1]/df[4,j+1]
            df[9,j+1] = df[1,j+1] - df[10,j+1]*df[3,j+1]
            upper_bound_DG = df[9,j+1] + band/2
            lower_bound_DG = df[9,j+1] - band/2
        
        df[11,j+1] = df[11,j]*np.exp(r*dt) - (df[7,j+1] - df[7,j])*df[0,j+1] - abs(df[7,j+1] - df[7,j])*scost
        
        df[12,j+1] = df[12,j]*np.exp(r*dt) - (df[9,j+1] - df[9,j])*df[0,j+1] - (df[10,j+1] - df[10,j])*df[6,j+1] - abs(df[9,j+1] - df[9,j])*scost - abs(df[10,j+1] - df[10,j])*ocost
        
    ### calculating pnl
    df[0,-1] = df[0,-2]*np.exp((mu-0.5*sigma*sigma)*dt + sigma*np.sqrt(dt)*np.random.randn()) 
    call_price, c_delta, c_gamma = Greeks_func(df[0,-1], K, r, T_s-T, sigma_q, otype='call')
    df[1,-1] = np.nan 
    df[2,-1] = np.nan  
    df[3,-1] = np.nan  
    df[4,-1] = np.nan  
    df[5,-1] = np.nan  
    df[6,-1] = call_price
    df[7,-1] = np.nan  
    df[8,-1] = np.nan  

    df[11,-1] = df[11,-2]*np.exp(r*dt) + df[7,-2]*df[0,-1] - np.maximum(K-df[0,-1],0) - abs(df[7,-2])*scost 
    df[10,-1] = np.nan
    df[9,-1] = np.nan
    df[12,-1] = df[12,-2]*np.exp(r*dt) + df[9,-2]*df[0,-1] + df[10,-2]*df[6,-1] - np.maximum(K-df[0,-1],0) - abs(df[9,-2])*scost - abs(df[10,-2])*ocost    
    return df



mu = 0.1
K = 100
r = 0.02
T = 0.25
T_s = 0.5
sigma = 0.2
sigma_q = 0.15
S0 = 100
scost = 0.005
ocost = 0.01



Sims = 1000
Sim_output = np.zeros((8,Sims))
for k in tqdm(range(Sims)):
    tb = time_based_h(mu, K, r, T, T_s, sigma, sigma, S0, scost, ocost, Nsteps=100)
    mb = move_based_h(mu, K, r, T, T_s, sigma, sigma, S0, scost, ocost, band=0.05, Nsteps=100)
    tb2 = time_based_h(mu, K, r, T, T_s, sigma, sigma_q, S0, scost, ocost, Nsteps=100)
    mb2 = move_based_h(mu, K, r, T, T_s, sigma, sigma_q, S0, scost, ocost, band=0.05, Nsteps=100)
    Sim_output[0,k] = tb[-2,-1] ### Delta Hedging
    Sim_output[1,k] = tb[-1,-1] ### Dleta-Gamma Hedging
    Sim_output[2,k] = mb[-2,-1]
    Sim_output[3,k] = mb[-1,-1]
    Sim_output[4,k] = tb2[-2,-1]
    Sim_output[5,k] = tb2[-1,-1]
    Sim_output[6,k] = mb2[-2,-1]
    Sim_output[7,k] = mb2[-1,-1]




Sim_mean = Sim_output.mean(axis=1)
result_df = pd.DataFrame(index=['time_based','move_based'],columns=['Delta','Delta-Gamma'])
vol_df = pd.DataFrame(index=['time_based','move_based'],columns=['Delta','Delta-Gamma'])
result_df.loc['time_based','Delta'] = Sim_mean[0]
result_df.loc['time_based','Delta-Gamma'] = Sim_mean[1]
result_df.loc['move_based','Delta'] = Sim_mean[2]
result_df.loc['move_based','Delta-Gamma'] = Sim_mean[3]
vol_df.loc['time_based','Delta'] = Sim_mean[4]
vol_df.loc['time_based','Delta-Gamma'] = Sim_mean[5]
vol_df.loc['move_based','Delta'] = Sim_mean[6]
vol_df.loc['move_based','Delta-Gamma'] = Sim_mean[7]




band_df = pd.DataFrame(index=range(1,51),columns=['Delta','Delta-Gamma'])
band_df.iloc[:,:] = 0
for b in tqdm(range(1,51)):
    mband = b/100
    for j in range(Sims):
        mb_test = move_based_h(mu, K, r, T, T_s, sigma, sigma, S0, scost, ocost, band=mband, Nsteps=100)
        band_df.iloc[b-1,0] += mb_test[-2,-1]
        band_df.iloc[b-1,1] += mb_test[-1,-1]
band_df = band_df/Sims
band_df.index = band_df.index/100



plt.figure(figsize=(12,8),dpi=100)
plt.plot(band_df.index,band_df)
plt.xlabel('width of the band',fontsize=15)
plt.ylabel('pnl',fontsize=15)
plt.title('the influence of the rebalancing-band on the hedge')
plt.show()




