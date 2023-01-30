#!/usr/bin/env python
# coding: utf-8

# # Water Quality Prediction

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)


# In[2]:


df = pd.read_csv('water_potability.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.describe()


# In[7]:


df.info


# In[8]:


df.nunique()


# In[9]:


df.isnull().sum()


# In[10]:


df.dtypes


# In[11]:


sns.heatmap(df.isnull())


# In[12]:


plt.figure(figsize = (10, 8))
sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')


# In[13]:


# Unstacking the correlation matrix to see the values more clearly.
corr = df.corr()
c1 = corr.abs().unstack()
c1.sort_values(ascending = False)[12:24:2]


# In[14]:


sns.countplot(x = 'Potability', data = df, saturation = 0.8)
plt.xticks(ticks = [0, 1], labels = ['Not Potable', 'Potable'])
plt.show()


# In[15]:


df.Potability.value_counts()


# In[16]:


sns.violinplot(x = 'Potability', y = 'ph', data = df, palette = 'rocket')


# In[17]:


plt.rcParams['figure.figsize'] = [20,10]
df.hist()
plt.show()


# **matplotlib.rcParams** contains some properties in matplotlibrc file. We can use it to control the defaults of almost every property in Matplotlib: figure size and DPI, line width, color and style, axes, axis and grid properties, text and font properties and so on.

# In[18]:


plt.figure(figsize = (20, 10))
df.hist()
plt.show()


# In[19]:


sns.pairplot(df, hue = 'Potability')


# In[20]:


plt.figure(figsize = (7, 5))
sns.distplot(df['Potability'])


# In[21]:


df.hist(column = 'ph', by = 'Potability')


# In[22]:


df.hist(column = 'Hardness', by = 'Potability')


# In[23]:


sns.histplot(x = 'Hardness', data = df)


# In[24]:


df.skew()


# In[25]:


skew_val = df.skew().sort_values(ascending=False)
skew_val


# - Using pandas skew function to check the correlation between the values.
# - Values between 0.5 to -0.5 will be considered as the normal distribution else will be skewed depending upon the skewness value.

# # Plotly

# ### Box Plot

# In[26]:


px.box(df, x=df.Potability, y=df.ph, color=df.Potability, width=600, height=400)


# In[27]:


px.box(df, x=df.Potability, y=df.Hardness, color=df.Potability, width=600, height=400)


# In[28]:


px.box(df, x=df.Potability, y=df.Solids, color=df.Potability, width=600, height=400)


# In[29]:


px.box(df, x=df.Potability, y=df.Chloramines, color=df.Potability, width=600, height=400)


# In[30]:


px.box(df, x=df.Potability, y=df.Sulfate, color=df.Potability, width=600, height=400)


# In[31]:


px.box(df, x=df.Potability, y=df.Conductivity, color=df.Potability, width=600, height=400)


# In[32]:


px.box(df, x=df.Potability, y=df.Organic_carbon, color=df.Potability, width=600, height=400)


# In[33]:


px.box(df, x=df.Potability, y=df.Trihalomethanes, color=df.Potability, width=600, height=400)


# In[34]:


px.box(df, x=df.Potability, y=df.Turbidity, color=df.Potability, width=600, height=400)


# ### Histogram

# In[35]:


px.histogram(df, x=df.ph, facet_row=df.Potability, template='plotly_dark')


# **facet_row (str or int or Series or array-like)** – Either a name of a column in data_frame, or a pandas Series or array_like object. Values from this column or array_like are used to assign marks to facetted subplots in the vertical direction.

# In[36]:


px.histogram(df, x=df.Sulfate, facet_row=df.Potability, template='plotly_white')


# In[37]:


px.histogram(df, x=df.Trihalomethanes, facet_row=df.Potability, template='ggplot2')


# ### Pie Chart

# In[38]:


px.pie(df, names=df.Potability, hole=0.4, template='plotly_dark')


# ### Scatter Plot

# In[39]:


px.scatter(df, x=df.ph, y=df.Sulfate, color=df.Potability, template='plotly_dark', trendline='ols')


# **Plotly Express** allows you to add **Ordinary Least Squares (OLS)** regression trendline to scatterplots with the trendline argument. In order to do so, you will need to install statsmodels and its dependencies. Hovering over the trendline will show the equation of the line and its R-squared value.

# In[40]:


px.scatter(df, x=df.Organic_carbon, y=df.Hardness, color=df.Potability, template='plotly_dark', trendline='ols')


# In[41]:


px.scatter(df, x=df.Chloramines, y=df.Turbidity, color=df.Potability, template='plotly_dark', trendline='ols')


# In[42]:


df.isnull().mean().plot.bar(figsize = (10, 6))
plt.title('Missing Data in Percentage')
plt.xlabel('Features')
plt.ylabel('Percentage of missing values')


# In[43]:


df.ph = df.ph.fillna(df.ph.mean())


# In[44]:


df.Sulfate = df.Sulfate.fillna(df.Sulfate.mean())


# In[45]:


df.Trihalomethanes = df.Trihalomethanes.fillna(df.Trihalomethanes.mean())


# In[46]:


df.head()


# In[47]:


df.isnull().sum()


# In[48]:


sns.heatmap(df.isnull())


# # Modeling

# In[49]:


from sklearn.model_selection import train_test_split


# In[50]:


X = df.drop('Potability', axis=1)
y = df.Potability

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=101)


# In[51]:


from sklearn.preprocessing import StandardScaler


# In[52]:


scaler = StandardScaler()


# In[53]:


X_train = scaler.fit_transform(X_train)
X_train


# ## Logistic Regression

# In[54]:


from sklearn.linear_model import LogisticRegression


# In[55]:


log_model = LogisticRegression()


# In[56]:


log_model.fit(X_train, y_train)


# In[57]:


pred_log = log_model.predict(X_test)


# In[58]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# In[59]:


lg = accuracy_score(y_test, pred_log)
print(lg)


# In[60]:


print(classification_report(y_test, pred_log))


# In[61]:


cm_log = confusion_matrix(y_test, pred_log)
ConfusionMatrixDisplay(confusion_matrix=cm_log).plot()


# ## Decision Tree Classifier

# In[62]:


from sklearn.tree import DecisionTreeClassifier


# In[63]:


dt_model = DecisionTreeClassifier()


# In[64]:


dt_model.fit(X_train, y_train)


# In[65]:


pred_dt = dt_model.predict(X_test)


# In[66]:


dt = accuracy_score(y_test, pred_dt)
print(dt)


# In[67]:


print(classification_report(y_test, pred_dt))


# In[68]:


cm_dt = confusion_matrix(y_test, pred_dt)
ConfusionMatrixDisplay(confusion_matrix=cm_dt).plot()


# ## Random Forest Classifier

# In[69]:


from sklearn.ensemble import RandomForestClassifier


# In[70]:


rf_model = RandomForestClassifier()


# In[71]:


rf_model.fit(X_train, y_train)


# In[72]:


pred_rf = rf_model.predict(X_test)


# In[73]:


rf = accuracy_score(y_test, pred_rf)
print(rf)


# In[74]:


print(classification_report(y_test, pred_rf))


# In[75]:


cm_rf = confusion_matrix(y_test, pred_rf)
ConfusionMatrixDisplay(confusion_matrix=cm_rf).plot()


# ## XGBoost Classifier

# In[76]:


from xgboost import XGBClassifier


# In[77]:


xgb_model = XGBClassifier()


# In[78]:


xgb_model.fit(X_train, y_train)


# In[79]:


pred_xgb = xgb_model.predict(X_test)


# In[80]:


xgb = accuracy_score(y_test, pred_xgb)
print(xgb)


# In[81]:


print(classification_report(y_test, pred_xgb))


# In[82]:


cm_xgb = confusion_matrix(y_test, pred_xgb)
ConfusionMatrixDisplay(confusion_matrix=cm_xgb).plot()


# ## K-Nearest Neighbours

# In[83]:


from sklearn.neighbors import KNeighborsClassifier


# In[84]:


knn_model = KNeighborsClassifier()


# In[85]:


knn_model.fit(X_train, y_train)


# In[86]:


pred_knn = knn_model.predict(X_test)


# In[87]:


knn = accuracy_score(y_test, pred_knn)
print(knn)


# In[88]:


print(classification_report(y_test, pred_knn))


# In[89]:


cm_knn = confusion_matrix(y_test, pred_knn)
ConfusionMatrixDisplay(confusion_matrix=cm_knn).plot()


# ## Support Vector Machine

# In[90]:


from sklearn.svm import SVC, LinearSVC


# In[91]:


svm_model = SVC()


# In[92]:


svm_model.fit(X_train, y_train)


# In[93]:


pred_svm = svm_model.predict(X_test)


# In[94]:


svm = accuracy_score(y_test, pred_svm)
print(svm)


# In[95]:


print(classification_report(y_test, pred_svm))


# In[96]:


cm_svm = confusion_matrix(y_test, pred_svm)
ConfusionMatrixDisplay(confusion_matrix=cm_svm).plot()


# ## AdaBoost Classifier

# In[97]:


from sklearn.ensemble import AdaBoostClassifier


# In[98]:


ada_model = AdaBoostClassifier(learning_rate= 0.002, n_estimators= 205)


# In[99]:


ada_model.fit(X_train, y_train)


# In[100]:


pred_ada = ada_model.predict(X_test)


# In[101]:


ada = accuracy_score(y_test, pred_ada)


# In[102]:


print(classification_report(y_test, pred_ada))


# In[103]:


cm_ada = confusion_matrix(y_test, pred_ada)
ConfusionMatrixDisplay(confusion_matrix=cm_ada).plot()


# In[104]:


models = pd.DataFrame({
    'Model':['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'KNeighbours', 'SVM', 'AdaBoost'],
    'Accuracy_score' :[lg, dt, rf, xgb, knn, svm, ada]
})
models


# In[105]:


sns.barplot(x='Accuracy_score', y='Model', data=models)


# Authored By:
# 
# **[Soumya Kushwaha](https://github.com/Soumya-Kushwaha)**
