#!/usr/bin/env python
# coding: utf-8

# # Advertisement Click Prediction
# 
# In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. 
# 
# We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('advertising.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# ## Exploratory Data Analysis

# In[6]:


sns.histplot(df['Age'], bins=30)


# In[7]:


sns.jointplot(data=df, x='Age', y='Area Income')


# In[8]:


sns.jointplot(data=df, x='Age', y='Daily Time Spent on Site', kind='kde')


# In[9]:


sns.jointplot(data=df, x='Daily Time Spent on Site', y='Daily Internet Usage')


# In[10]:


sns.pairplot(data=df, hue='Clicked on Ad')


# # Logistic Regression

# In[11]:


from sklearn.model_selection import train_test_split

X = df.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis=1)
Y = df['Clicked on Ad']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# In[12]:


from sklearn.linear_model import LogisticRegression


# In[13]:


log = LogisticRegression(max_iter = 200)


# In[14]:


log.fit(X_train, Y_train)


# ## Predictions and Evaluations

# In[15]:


predictions = log.predict(X_test)


# In[16]:


from sklearn.metrics import classification_report, confusion_matrix


# In[17]:


print(classification_report(Y_test, predictions))


# In[18]:


print(confusion_matrix(Y_test, predictions))


# ### Authored by: 
# [Soumya Kushwaha](https://github.com/Soumya-Kushwaha)
