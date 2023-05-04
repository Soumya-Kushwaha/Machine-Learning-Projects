#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_excel('Data_Train.xlsx')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data['Airline'].unique()


# In[7]:


data.isnull().sum()


# In[8]:


data.dropna(inplace = True)


# In[9]:


data['Date_of_Journey'] = pd.to_datetime(data.Date_of_Journey, infer_datetime_format=True)


# In[10]:


data['Day'] = pd.DatetimeIndex(data.Date_of_Journey).day


# In[11]:


data['Month'] = pd.DatetimeIndex(data.Date_of_Journey).month


# In[12]:


data['Year'] = pd.DatetimeIndex(data.Date_of_Journey).year


# In[13]:


data.head()


# In[14]:


data.drop('Date_of_Journey', axis=1, inplace=True)


# In[15]:


data['Dep_Hour'] = pd.to_datetime(data.Dep_Time).dt.hour


# In[16]:


data['Dep_Min'] = pd.to_datetime(data.Dep_Time).dt.minute


# In[17]:


data.drop('Dep_Time', axis=1, inplace=True)


# In[18]:


data['Arr_Hour'] = pd.to_datetime(data.Arrival_Time).dt.hour


# In[19]:


data['Arr_Min'] = pd.to_datetime(data.Arrival_Time).dt.minute


# In[20]:


data.drop('Arrival_Time', axis=1, inplace=True)


# In[21]:


data.head()


# In[22]:


data.shape


# In[23]:


data.Total_Stops.unique()


# In[24]:


data.Total_Stops.value_counts()


# In[25]:


data[data.Total_Stops.isnull()]


# In[26]:


data['Total_Stops']=data['Total_Stops'].map({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4,'nan':1})


# In[27]:


data.head()


# In[28]:


# Time taken by plane to reach destination is called Duration
# It is the differnce betwwen Departure Time and Arrival time


# Assigning and converting Duration column into list
duration = list(data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extra


# In[29]:


data['Duration_Hours'] = duration_hours
data['Duration_Mins'] = duration_mins


# In[30]:


data.drop(['Duration'], axis=1, inplace=True)


# In[31]:


data.head()


# In[32]:


data.info()


# ## Handling Categorical Data
# One can find many ways to handle categorical data. Some of them categorical data are,
# 
# - **Nominal data** --> data are not in any order --> OneHotEncoder is used in this case
# - **Ordinal data** --> data are in order --> LabelEncoder is used in this case

# **● Airline**

# In[33]:


data.Airline.unique()


# In[34]:


data.Airline.value_counts()


# In[35]:


# Airplane vs Price

sns.catplot(x = "Airline", y = 'Price', data = data.sort_values("Price", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()


# From graph we can see that **Jet Airways** airlines have the highest price.<br>Apart from the first airline almost all are having similar median.

# In[36]:


# As Airline is Nominal Categorical data we will perform OneHotEncoding:

Airline = data[['Airline']]
Airline = pd.get_dummies(Airline, drop_first=True)
Airline.head()


# In[37]:


data.groupby('Airline')['Price'].mean().sort_values()


# **● Source**

# In[38]:


data.Source.value_counts()


# In[39]:


# Source vs Price

sns.catplot(x = "Source", y = "Price", data = data.sort_values("Price", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()


# In[40]:


# As Source is Nominal Categorical data we will perform OneHotEncoding

Source = data[['Source']]
Source = pd.get_dummies(Source, drop_first = True)
Source.head()


# **● Destination**

# In[41]:


data.Destination.value_counts()


# In[42]:


# As Destination is Nominal Categorical data we will perform OneHotEncoding

Destination = data[['Destination']]
Destination = pd.get_dummies(Destination, drop_first = True)
Destination.head()


# **● Route**

# In[43]:


data.Route.value_counts()


# In[44]:


# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other

data.drop(['Route', 'Additional_Info'], axis=1, inplace=True)


# **● Total Stops**

# In[45]:


data.Total_Stops.value_counts()


# In[46]:


# As this is case of Ordinal Categorical type we perform LabelEncoder
# Here Values are assigned with corresponding keys

data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace=True)
data.head()


# In[47]:


# Concatenate dataframe --> train_data + Airline + Source + Destination

data = pd.concat([data, Airline, Source, Destination], axis = 1)
data.head()


# In[48]:


data.shape


# ## One Hot Encoder

# In[49]:


from sklearn.preprocessing import OneHotEncoder


# In[50]:


ohe = OneHotEncoder()


# In[51]:


ohe.fit_transform(data[['Airline']]).toarray()


# In[52]:


## Replacing target guided ordinal encoding

def replace_airline_with_mean(df):
    mean_prices = data.groupby('Airline')['Price'].mean().sort_values()
    data['Airline'] = data['Airline'].apply(lambda x: mean_prices[x])
    return df

data = replace_airline_with_mean(data)
data.head()


# In[53]:


pd.DataFrame(ohe.fit_transform(data[['Airline']]).toarray(),columns=ohe.get_feature_names_out())


# ## Test Set

# In[54]:


test = pd.read_excel('Test_set.xlsx')
test.head()


# In[55]:


# Preprocessing

print("Test Info")
print("-"*75)
print(test.info())

print()
print()

print("Null values: ")
print("-"*75)
print(test.isnull().sum())


# In[56]:


# Feature Engineering

## Date of Journey
test['Date_of_Journey'] = pd.to_datetime(test.Date_of_Journey, infer_datetime_format=True)
test['Day'] = pd.DatetimeIndex(test.Date_of_Journey).day
test['Month'] = pd.DatetimeIndex(test.Date_of_Journey).month
test['Year'] = pd.DatetimeIndex(test.Date_of_Journey).year
test.drop(['Date_of_Journey'], axis=1, inplace=True)


# In[57]:


# Departure Time
test['Dep_Hour'] = pd.to_datetime(test.Dep_Time).dt.hour
test['Dep_Min'] = pd.to_datetime(test.Dep_Time).dt.minute
test.drop(['Dep_Time'], axis=1, inplace=True)


# In[58]:


# Arrival Time
test['Arr_Hour'] = pd.to_datetime(test.Arrival_Time).dt.hour
test['Arr_Min'] = pd.to_datetime(test.Arrival_Time).dt.minute
test.drop(['Arrival_Time'], axis=1, inplace=True)


# In[59]:


# Duration
duration = list(test["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding Duration column to test set
test["Duration_hours"] = duration_hours
test["Duration_mins"] = duration_mins
test.drop(["Duration"], axis = 1, inplace = True)


# In[60]:


# Categorical data

print("Airline")
print("-"*75)
print(test["Airline"].value_counts())
Airline = pd.get_dummies(test["Airline"], drop_first= True)

print()

print("Source")
print("-"*75)
print(test["Source"].value_counts())
Source = pd.get_dummies(test["Source"], drop_first= True)

print()

print("Destination")
print("-"*75)
print(test["Destination"].value_counts())
Destination = pd.get_dummies(test["Destination"], drop_first = True)


# In[61]:


# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other
test.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

# Replacing Total_Stops
test.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

# Concatenate dataframe --> test_data + Airline + Source + Destination
test = pd.concat([test, Airline, Source, Destination], axis = 1)


# In[62]:


print("Shape of test data : ", test.shape)


# In[63]:


test.head()


# In[ ]:




