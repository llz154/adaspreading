#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import numpy as np
import pandas as pd
import random
import copy
import math
from collections import OrderedDict, Counter
import os,time
import itertools
import matplotlib.pyplot as plt
from multiprocessing import Pool   #python环境
#from multiprocess import Pool       #jupyter环境
#import hypernetx as hnx
import json


# In[2]:


_Data_PATH_XX_ = './simulation_results/'
_Figure_PATH_XX_ = './figures/'


# In[3]:


#生成m-uniform-hypergraph
N = 10000
Edge = 20000
beta = 0.04
r = 0.1 #重连指数函数的基底
v = 2.4  #非线性传播指数
h = 4.5 #非线性重连指数
m = 3  #超边大小
mu = 0.2 #恢复概率
lam = 1 # 连接S的偏好或准确程度，对应方程中的lam参数(之前为参数p)
n0= 300
T = 1000
repeated_time = 30
beta_list = []
a1 = 7
a2 = 10
for i in range(21):
    if i <= a1:
        beta_list.append(0.01 + i*0.002)
    elif i > a1 and i <= a2:
        beta_list.append(0.01 + (i-1)*0.002+0.001)
        beta_list.append(0.01 + i*0.002)
    else:
        beta_list.append(0.01 + i*0.002)


# In[4]:


filename = 'testN'+str(N)+'Edge' + str(Edge) + '_v'+ str(v) + '_r' + str(r) + '_h' + str(h) + '_p' +str(lam) +'_mu'+str(mu)+'n0'+str(n0)


# In[5]:


def Generating_uniform_Hyper(N,Edge,m):
    edge_list = []
    edge_set = set()
    count = 0
    while count < Edge:
        e = np.random.randint(0,N,size = m)
        if len(set(e)) == m:
            if frozenset(e) in edge_set:
                continue
            else:
                edge_list.append(list(e))
                edge_set.add(frozenset(e))
                count = count + 1
        else:
            continue
    return edge_list

edge_list = Generating_uniform_Hyper(N,Edge,m)


# In[6]:


# 节点初始状态
def initial_state(N,n0,edge_list):
    if n0 > 2400:
        inf_set = set([i for i in range(N)])
        sus_set = set()
        i = 0
        while i<N-n0:
            k = np.random.randint(N)
            while k in sus_set:
                k = np.random.randint(N)
            sus_set.add(k)
            inf_set.remove(k)
            i = i + 1
    else:
        inf_set = set()
        sus_set = set([i for i in range(N)])
        i = 0
        while i< n0:
            k = np.random.randint(N)
            while k in inf_set:
                k = np.random.randint(N)
            inf_set.add(k)
            sus_set.remove(k)
            i=i+1
            
    #初始时点—边字典：key是节点编号，value是连边集合
    node_edge_dict={}
    for i in range(N):
        node_edge_dict[i]=set()
    for edge in edge_list:
        for j in edge:
            node_edge_dict[j].add(frozenset(edge))

    # 记录7种状态的连边：
    #hyper_list包含四个连边集合：没有感染节点、1个感染节点、2个感染节点，3个感染节点
    hyper_list = [set(),set(),set(),set()]
    
    for edge in edge_list:
        temp = 0
        for j in edge:
            if j in inf_set:
                temp = temp + 1
            
        hyper_list[temp].add(frozenset(edge))
            
    return inf_set, sus_set,hyper_list, node_edge_dict


# In[7]:


#定义重连偏好
def S_preference(I_num,S_num,lam):
    return (lam*S_num)/(lam*S_num+I_num)


# In[8]:


#Gillesipe算法实现：涉及超边传播速率（n-i）*beta * (infected_nodes**v)、超边重连速率r*(infected_nodes**h)、恢复速率mu
#定义超边传播速率发生带来的改变
                
def hyper_31_spreading(inf_set, sus_set, hyper_list, node_edge_dict):

    edge_of_spreading = random.choice(list(hyper_list[1]))
    temp = 0 # 记录状态改变的节点标号
    sus_subset = set()
    for j in edge_of_spreading:
        if j in sus_set:
            sus_subset.add(j)
        
    temp = random.choice(list(sus_subset)) 
        
    # 节点状态变化
    inf_set.add(temp)
    sus_set.remove(temp)
        
    #连边状态变化
    for edge in node_edge_dict[temp]:
        tem = 0
        for j in edge:
            if j in inf_set:
                tem = tem +1
        hyper_list[tem].add(edge)
        hyper_list[tem-1].remove(edge)
    return inf_set, sus_set, hyper_list

def hyper_32_spreading(inf_set, sus_set,hyper_list, node_edge_dict):

    edge_of_spreading = random.choice(list(hyper_list[2]))
    temp = 0 # 记录状态改变的节点标号
    for j in edge_of_spreading:
        if j in sus_set:
            temp = j
        
    # 节点状态变化
    inf_set.add(temp)
    sus_set.remove(temp)
        
    #连边状态变化
    for edge in node_edge_dict[temp]:
        tem = 0
        for j in edge:
            if j in inf_set:
                tem = tem +1
        hyper_list[tem].add(edge)
        hyper_list[tem-1].remove(edge)
    return inf_set, sus_set, hyper_list
    
def hyper31_rewiring(inf_set, sus_set, hyper_list, node_edge_dict,lam):
    edge_of_rewiring = random.choice(list(hyper_list[1]))
    temp = 0 # 记录参与重连的点
    sus_subset = set()
    for j in edge_of_rewiring:
        if j in sus_set:
            sus_subset.add(j)
    temp = random.choice(list(sus_subset))
    
    hyper_list[1].remove(edge_of_rewiring) #连边断开
    for j in edge_of_rewiring:
        node_edge_dict[j].remove(edge_of_rewiring)
    
    I_num = len(inf_set)
    S_num = len(sus_set)
    prob_of_S = S_preference(I_num,S_num,lam)
    
    #选择两个点
    i = 0
    inf_local = 0
    rewiring_set = set()
    rewiring_set.add(temp)
    while i < 2:
        if random.random() < prob_of_S:
            tem = random.choice(list(sus_set))
            while tem in rewiring_set:
                tem = random.choice(list(sus_set))
            
            rewiring_set.add(tem)
        else:
            tem = random.choice(list(inf_set))
            while tem in rewiring_set:
                tem = random.choice(list(inf_set))
            rewiring_set.add(tem)
            inf_local = inf_local + 1
        
        i = i + 1
        
    new_edge = list(rewiring_set)
    hyper_list[inf_local].add(frozenset(new_edge))
    for j in new_edge:
        node_edge_dict[j].add(frozenset(new_edge))
    return inf_set, sus_set, hyper_list, node_edge_dict

def hyper32_rewiring(inf_set, sus_set, hyper_list, node_edge_dict,lam):
    edge_of_rewiring = random.choice(list(hyper_list[2]))
    temp = 0 # 记录参与重连的点
    for j in edge_of_rewiring:
        if j in sus_set:
            temp = j
            break
    
    hyper_list[2].remove(edge_of_rewiring) #连边断开
    for j in edge_of_rewiring:
        node_edge_dict[j].remove(edge_of_rewiring)
    
    I_num = len(inf_set)
    S_num = len(sus_set)
    prob_of_S = S_preference(I_num,S_num,lam)
    
    #选择两个点
    i = 0
    inf_local = 0
    rewiring_set = set()
    rewiring_set.add(temp)
    while i < 2:
        if random.random() < prob_of_S:
            tem = random.choice(list(sus_set))
            while tem in rewiring_set:
                tem = random.choice(list(sus_set))
            
            rewiring_set.add(tem)
        else:
            tem = random.choice(list(inf_set))
            while tem in rewiring_set:
                tem = random.choice(list(inf_set))
            rewiring_set.add(tem)
            inf_local = inf_local + 1
        
        i = i + 1
        
    new_edge = list(rewiring_set)
    hyper_list[inf_local].add(frozenset(new_edge))
    for j in new_edge:
        node_edge_dict[j].add(frozenset(new_edge))
    return inf_set, sus_set, hyper_list, node_edge_dict

def recover(inf_set, sus_set, hyper_list, node_edge_dict):
    rec_node = random.choice(list(inf_set))
    
    inf_set.remove(rec_node)
    sus_set.add(rec_node)
    
    #连边更新
    for edge in node_edge_dict[rec_node]:
        tem = 0
        for j in edge:
            if j in inf_set:
                tem = tem +1
        hyper_list[tem].add(edge)
        hyper_list[tem+1].remove(edge) 
    
    return inf_set, sus_set, hyper_list


# In[9]:


#定义当前状态下7类过程的速率
def dynamic_rate(inf_set, hyper_list, beta, r, v, h, mu):
    rate_recover = mu * len(inf_set)
    rate_hyper31_s = 2 * beta * len(hyper_list[1])
    rate_hyper32_s = beta * (2**v) * len(hyper_list[2])
    rate_hyper31_r = r * len(hyper_list[1])
    rate_hyper32_r = r * (2**h) * len(hyper_list[2])
    
    return rate_recover, rate_hyper31_s, rate_hyper32_s, rate_hyper31_r, rate_hyper32_r


# In[10]:


def Gillespie_simulation(args):
    
    it_num, N, edge_list, beta,r,v,h,mu,lam,T,n0 = args
    
    inf_set, sus_set, hyper_list, node_edge_dict = initial_state(N,n0,edge_list)
    t = 0
    inf_n = 0
    i_count = 0
    while t < T:
        r1 = random.random()
        r2 = random.random()
        
        rate_recover, rate_hyper31_s, rate_hyper32_s, rate_hyper31_r, rate_hyper32_r = dynamic_rate(inf_set, hyper_list, beta, r, v, h, mu)
        rate_sum = rate_recover+rate_hyper31_s+rate_hyper32_s+rate_hyper31_r+rate_hyper32_r
        
        
        if rate_sum == 0:
            break
        
        tau = -math.log(r1) / rate_sum
        
        if r2 * rate_sum <  rate_recover:
            inf_set, sus_set, hyper_list = recover(inf_set, sus_set, hyper_list, node_edge_dict)
        elif r2 * rate_sum < rate_recover + rate_hyper31_s:
            inf_set, sus_set, hyper_list = hyper_31_spreading(inf_set, sus_set, hyper_list, node_edge_dict)
        elif r2 * rate_sum < rate_recover + rate_hyper31_s + rate_hyper32_s:
            inf_set, sus_set, hyper_list = hyper_32_spreading(inf_set, sus_set, hyper_list, node_edge_dict)
        elif r2 * rate_sum < rate_recover + rate_hyper31_s + rate_hyper32_s + rate_hyper31_r:
            inf_set, sus_set, hyper_list, node_edge_dict = hyper31_rewiring(inf_set, sus_set, hyper_list, node_edge_dict,lam)
        else:
            inf_set, sus_set, hyper_list, node_edge_dict = hyper32_rewiring(inf_set, sus_set, hyper_list, node_edge_dict,lam)
        
        t = t + tau
        

        if t > T-50:
            inf_n = inf_n + len(inf_set)
            i_count = i_count + 1
        
    if i_count == 0 & inf_n == 0:
        inf_ave = 0
    else:
        inf_ave = inf_n / i_count
    return inf_ave     


# In[11]:


# 定义多次仿真的动力学过程
"""def repeated_simulation(N,beta,r,v,h,mu,lam,T,n0,repeated_time,edge_list):
    infected_num = []
    for i in range(repeated_time):
        inf_n, inf_set, sus_set, hyper_list, node_edge_dict = Gillespie_simulation(N, edge_list,beta,r,v,h,mu,lam,T,n0)
        infected_num.append(inf_n)
    
    return infected_num"""


# In[ ]:


#定义并行
if __name__ == "__main__":
    rho_I_dict = {}
    for beta in beta_list:
        infected_num = []
        args=[]
        for it_num in range(repeated_time):
            args.append([it_num, N, edge_list, beta, r, v, h, mu, lam, T, n0])
        
        with Pool(10) as pool:
            results = pool.map(Gillespie_simulation, args)
        for res in results:
            infected_num.append(res)
            
        rho_I_dict[beta] = infected_num
        #print(infected_num)
    
    np.save(_Data_PATH_XX_ + filename + '.npy', rho_I_dict)


# In[13]:


#数据存储

