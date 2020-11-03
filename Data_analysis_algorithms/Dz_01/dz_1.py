#!/usr/bin/env python
# coding: utf-8

# ### 1. Подберите скорость обучения (alpha) и количество итераций для достижения минимальног значения функции потерь; ###
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

y = [45, 55, 50, 59, 65, 35, 75, 80, 50, 60]

X = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 2, 1, 3, 0, 5, 10, 1, 2]])
X.shape


# In[2]:


def calc_mse(y, y_pred):
    err = np.mean((y - y_pred)**2)
    return err


# In[44]:


n = X.shape[1]
alpha = 0.063
iteration = 293
W = np.array([1, 0.5])
W, alpha


# In[46]:


for i in range(iteration):
    y_pred = np.dot(W, X)
    err = calc_mse(y, y_pred)
    spam = err
    for ii in range(W.shape[0]):
        W[ii] -= alpha * (1/n * 2 * np.sum(X[ii] * (y_pred - y)))
#         if i % 100 == 0:
print(i, W, err)


# ### 2 (опция). В этом коде мы избавляемся от итераций по весам, но тут есть ошибка, исправьте ее: ###
# 

# In[30]:


for i in range(iteration):
    y_pred = np.dot(W, X)
    err = calc_mse(y, y_pred)
    W -= alpha * (1/n * 2 * np.dot(X, (y_pred - y)))
    if i % 100 == 0:
        print(i, W, err)
print(i, W, err)


# ### 3 (опция). Реализовать один из критериев останова, перечисленный в методичке.##

# In[32]:


i = 0
while True:
    i+=1
    y_pred = np.dot(W, X)
    err = calc_mse(y, y_pred)
    W_0 = W.copy()
    W -= alpha * (1/n * 2 * np.dot(X, (y_pred - y)))
    if np.sum(np.abs(W-W_0)) < 0.00000001:
        break
#     if i % 100 == 0:
#         print(i, W, err)
print((i, W, err))


# __***Подбор $\alpha$ и количество итераций__

# In[41]:


i_old = 10**20
for alpha in np.linspace(1e-3, 1, num = 1000):
    n = X.shape[1]
    W = np.array([1, 0.5])
    i = 0
    while True:
        i+=1
        y_pred = np.dot(W, X)
        err = calc_mse(y, y_pred)
        W_0 = W.copy()
        W -= alpha * (1/n * 2 * np.dot(X, (y_pred - y)))
        if np.sum(np.abs(W-W_0)) < 0.00000001:
            break
    if i_old < i:
        break
    i_old = i
    print((i, W, alpha, err))
    


# In[ ]:




