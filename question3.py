#!/usr/bin/env python
# coding: utf-8

### pricing project 1, question 3
import pandas as pd
import numpy as np
import datetime as dt
#from scipy import linalg
import matplotlib.pyplot as plt
from tqdm import tqdm  #### loop 
from sklearn.neighbors import KernelDensity
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def boundary_func(T=1, S0=10, K=10, mu=0.05, sigma=0.2, r=0.02, N=5000):
    dt = T/N
    q = (1-np.exp(-sigma*np.sqrt(dt)))/(np.exp(sigma*np.sqrt(dt))-np.exp(-sigma*np.sqrt(dt)))
    u = np.exp(r*dt + sigma*np.sqrt(dt))
    d = np.exp(r*dt - sigma*np.sqrt(dt))
    
    S_table = np.zeros((N+1,N+1))
    P_table = np.zeros((N+1,N+1))
    S_table[0,0] = S0
    for k in tqdm(range(1,N+1)):
        S_table[:,k] = S_table[:,k-1]*u
        S_table[k,k] = S_table[k-1,k-1]*d
    P_table[:,N] = np.maximum(K-S_table[:,N],0)
    
    i_table = np.zeros((N+1,N+1))
    for k in tqdm(range(1,N+1)):
        P_table[:N-k+1,N-k] =  (P_table[:N-k+1,N-k+1]*q + P_table[1:N-k+2,N-k+1]*(1-q))/np.exp(r*dt)
        i_table[:N-k+1,N-k] = np.where(np.maximum(K - S_table[:N-k+1,N-k],0) > P_table[:N-k+1,N-k],1,0)
        P_table[:N-k+1,N-k] = np.maximum(np.maximum(K - S_table[:N-k+1,N-k],0),P_table[:N-k+1,N-k])
    P_table1 = np.multiply(i_table,S_table)
    P_boundary = np.max(P_table1,axis=0)
    boundary_df = pd.DataFrame(index=np.array(range(N+1))/N,columns=['boundary'])
    boundary_df['boundary'] = P_boundary
    boundary_df.loc[1.0,'boundary'] = K
    put_price = P_table[0,0]
    return put_price,boundary_df



def SimPrice(put_price,boundary_df,T=1, S0=10, K=10, mu=0.05, sigma=0.2, r=0.02, N=5000, Nsims=10000):
    pnl = np.zeros(Nsims)
    tlist = np.zeros(Nsims)
    t = np.linspace(0,T,N+1)
    dt = t[1] - t[0]
    p = 0.5*(1 + ((mu-r) - 0.5*sigma*sigma)*np.sqrt(dt)/sigma)
    
    S = np.zeros((Nsims,len(t)))
    S[:,0] = S0
    for i in tqdm(range(len(t)-1)):
        U = np.random.rand(Nsims)
        up = (U < p)
        x = 2*up - 1
        S[:,i+1] = S[:,i]*np.exp(r*dt + sigma*np.sqrt(dt)*x)
        
    idd = 0
    for i in range(Nsims):
        e_idx = np.where(S[i,:] <= np.array(boundary_df.boundary),1,0)
        if (e_idx==0).all():
            idd += 1
            pnl[i] = -put_price
            tlist[i] = -1
        else:
            e_idxdf = pd.Series(e_idx,index=range(N+1))
            e_idx1 = e_idxdf[e_idxdf>0].index[0]
            tlist[i] = e_idx1*dt
            pnl[i] = (K - S[i,e_idx1])/np.exp(r*tlist[i]) - put_price
    
    print(idd)         
    return pnl,tlist



##calculation for put price and boundary
put_price1,boundary_df1 = boundary_func(T=1, S0=10, K=10,mu=0.05, sigma=0.2, r=0.02, N=5000)
put_price2,boundary_df2 = boundary_func(T=1, S0=10, K=10,mu=0.05, sigma=0.1, r=0.02, N=5000)
put_price3,boundary_df3 = boundary_func(T=1, S0=10, K=10,mu=0.05, sigma=0.2, r=0.04, N=5000)
put_price4,boundary_df4 = boundary_func(T=1, S0=10, K=10,mu=0.05, sigma=0.1, r=0.04, N=5000)


pnl,tlist = SimPrice(put_price=put_price1, boundary_df=boundary_df1, T=1, S0=10, K=10, mu=0.05, sigma=0.2, r=0.02, N=5000, Nsims=10000)

### kernel density estimation of pnl
plt.figure(figsize=(12,8),dpi=100)
sns.distplot(pnl[pnl>-put_price1])
##sns.distplot(pnl)
plt.show()

print(pnl.mean())


### kernel density estimation of exercise time
plt.figure(figsize=(12,8),dpi=100)
sns.distplot(tlist[tlist>0])
plt.show()

print(tlist[tlist>0].mean())


### simulation under different realized volatility
pnl2,tlist2 = SimPrice(put_price=put_price1, boundary_df=boundary_df1, T=1, S0=10, K=10, mu=0.05, sigma=0.10, r=0.02, N=5000, Nsims=10000)
pnl3,tlist3 = SimPrice(put_price=put_price1, boundary_df=boundary_df1, T=1, S0=10, K=10, mu=0.05, sigma=0.15, r=0.02, N=5000, Nsims=10000)
pnl4,tlist4 = SimPrice(put_price=put_price1, boundary_df=boundary_df1, T=1, S0=10, K=10, mu=0.05, sigma=0.20, r=0.02, N=5000, Nsims=10000)
pnl5,tlist5 = SimPrice(put_price=put_price1, boundary_df=boundary_df1, T=1, S0=10, K=10, mu=0.05, sigma=0.25, r=0.02, N=5000, Nsims=10000)
pnl6,tlist6 = SimPrice(put_price=put_price1, boundary_df=boundary_df1, T=1, S0=10, K=10, mu=0.05, sigma=0.30, r=0.02, N=5000, Nsims=10000)



plt.figure(figsize=(12,8),dpi=100)
sns.distplot(pnl2[pnl2>-put_price1])
plt.show()


print(pnl2.mean())

plt.figure(figsize=(12,8),dpi=100)
sns.distplot(tlist2[tlist2>0])
plt.show()

print(tlist2[tlist2>0].mean())


##### boundary curve
plt.figure(figsize=(12,8),dpi=300)
plt.plot(boundary_df1.index,boundary_df1,linewidth=1,label='$\sigma$=0.2,r=0.02')
plt.plot(boundary_df2.index,boundary_df2,linewidth=1,label='$\sigma$=0.1,r=0.02')
plt.plot(boundary_df3.index,boundary_df3,linewidth=1,label='$\sigma$=0.2,r=0.04')
plt.plot(boundary_df4.index,boundary_df4,linewidth=1,label='$\sigma$=0.1,r=0.04')
plt.xlabel(r'$t$',fontsize=15)
plt.ylabel(r'$S_t$',fontsize=15)
plt.title('exercise boundary',fontsize=15)
plt.legend(loc=0,ncol=1,fontsize=15)
plt.show()






