#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


dataset = pd.read_csv("homeprices.csv")


# In[5]:


X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values


# In[6]:



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=1/3,random_state=0)


# In[7]:



from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)


# In[10]:


plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Area vs Price (Training Set)')
plt.xlabel('Price')
plt.ylabel('Area')
plt.show()


# In[11]:


plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title('Area vs Price (Test Set)')
plt.xlabel('Price')
plt.ylabel('Area')
plt.show()


# In[ ]:




