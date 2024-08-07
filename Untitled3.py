#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math 
def basic_sigmoid(x):
  s = 1 / (1 + math.exp(-x))
  return s#image2vector
def image2vector(image):
  v = image.reshape((v.shape[0]*v.shape[1],v.shape[2]))
  return v


# In[2]:


basic_sigmoid(-1)


# In[4]:


# using vectors(input)
import numpy as np

x = np.array([1,2,3])
print(np.exp(x))


# In[5]:


x = np.array([1,2,3])
print(x+3)


# In[6]:


import numpy as np

def sigmoid(x):
    s = 1/(1 + np.exp(-x))
    return s


# In[8]:


s = np.array([1,2,3])
sigmoid(x)


# In[16]:


#computing sigmoid gradient
def sigmoid_derivatives(x):
     
    s = 1 / (1+np.exp(-x))
    ds = s * (1-s)

    return ds


# In[22]:


x = np.array([1,2,3])
print("sigmoid_derivatives(x) = "+str(sigmoid_derivatives(x)))


# In[23]:


#image2vector
def image2vector(image):
  v = image.reshape((v.shape[0]*v.shape[1],v.shape[2]))
  return v


# In[25]:


import numpy as np

image = np.array([[[0.67826139, 0.29380381],
                   [0.90714982, 0.52835647],
                   [0.4215251, 0.45017551]],
                  
                  [[0.92814219, 0.96677647],
                   [0.85304703, 0.52351845],
                   [0.19981397, 0.27417313]],
                  
                  [[0.60659855, 0.00533165],
                   [0.10820313, 0.49978937],
                   [0.34144279, 0.94630077]]])

def image2vector(image):
    return image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))

print("image2vector(image) = " + str(image2vector(image)))


# In[ ]:


#normalization1


# In[26]:


def normalized_row(x):
  x_norm = np.linalg.norm(x,axis=1,keepdims=True)
  x = x / x_norm
  return x


# In[29]:


x= np.array([
    [0,1000,4],
    [2,6,4]])
#print normalized_row(x)
print("normalized_row(x) = " + str(normalized_row(x)))


# In[30]:


def softmax(x):
  x_exp = np.exp(x)
  x_sum = np.sum(x_exp,axis=1,keepdims=True)
  s = x_exp / x_sum
  return s


# In[31]:


x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))


# In[33]:


from typing import TypedDict
import time 
x1 = [9, 2, 5, 0, 0,7,5,0,0,0,9,2,5,0,0]
x2 = [9, 2, 2, 9, 0,9,2,5,0,0,9,2,5,0,0]
#dot product of vectors
tic = time.process_time()
dot = 0
for i in range(len(x1)):
  dot += x1[i] * x2[i]
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")


# In[34]:


import time
import numpy as np

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

# Outer product of vectors
tic = time.process_time()
outer_product = np.zeros((len(x1), len(x2)))
for i in range(len(x1)):
    for j in range(len(x2)):
        outer_product[i, j] = x1[i] * x2[j]
toc = time.process_time()

print("outer_product = \n" + str(outer_product))
print("----- Computation time = " + str(1000 * (toc - tic)) + "ms")


# In[35]:


import numpy as np
import time

x1 = np.array([9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0])
x2 = np.array([9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0])

# Elementwise multiplication and sum (dot product)
tic = time.process_time()
dot = np.dot(x1, x2)
toc = time.process_time()

print("dot = " + str(dot) + "\n----- Computation time = " + str(1000 * (toc - tic)) + "ms")


# In[ ]:




