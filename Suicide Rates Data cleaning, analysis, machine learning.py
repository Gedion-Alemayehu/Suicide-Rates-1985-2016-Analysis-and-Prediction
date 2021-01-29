#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import numpy as np     
import pandas as pd
import matplotlib.pyplot as plt
# Seaborn visualization library
import seaborn as sns
import statsmodels.api as sm


# ###### Read our suicide rate data set 

# In[50]:


df = pd.read_csv('master.csv')
df


# ###### Clean the data

# In[3]:


# change the columns names for better description and easy use
df.columns=["country","year","gender","age_group","number_of_suicide","population","suicide_per_100_thousand","country_year","hdi","gdp","gdp_per_capita","generation"]
df.head()


# In[4]:


# since hdi has a lot of missing values, we can replace it with the mean
df.fillna(df.mean(), inplace=True)
df


# In[5]:


# delete column 'country_year' for concision
del df['country_year']
#delete hdi because of a lot of missing data
#del df['hdi']
df.head()


# In[6]:


# change the age group to text description
df.age_group.unique()


# In[7]:


len(df.country.unique())


# In[9]:


df.gender.unique()


# In[10]:


df["age_group"]=df["age_group"].apply(lambda x: str(x).replace('5-14 years','9.5') if '5-14 years' in str(x) else str(x))
df["age_group"]=df["age_group"].apply(lambda x: str(x).replace('15-24 years','19.5') if '15-24 years' in str(x) else str(x))
df["age_group"]=df["age_group"].apply(lambda x: str(x).replace('25-34 years','29.5') if '25-34 years' in str(x) else str(x))
df["age_group"]=df["age_group"].apply(lambda x: str(x).replace('35-54 years','44.5') if '35-54 years' in str(x) else str(x))
df["age_group"]=df["age_group"].apply(lambda x: str(x).replace('55-74 years','64.5') if '55-74 years' in str(x) else str(x))
df["age_group"]=df["age_group"].apply(lambda x: str(x).replace('75+ years','75') if '75+ years' in str(x) else str(x))
df.head()


# In[11]:


df['age_group'] = df['age_group'].astype(float)


# In[12]:


lst = []
for i in range(len(df.gender)):
    if df.gender[i] == 'male':
        lst.append(0)
    else:
        lst.append(1)
df['sex'] = lst
df.head()


# In[13]:


df


# ###### Data exploration and summary 

# In[14]:


# Create the default pairplot
#sns.pairplot(df)


# ##### Suicide number by Year

# In[15]:


plt = plt.subplots(1, 1, figsize = (16 ,6))
plt = sns.barplot(x = df['year'], y = 'number_of_suicide',data = df, palette='Spectral')


# In[16]:


# arrange number of suicided from high to low
suicide_year = df.groupby('year')[['number_of_suicide']].sum().reset_index()
suicide_year.sort_values(by='number_of_suicide', ascending=False).style.background_gradient(cmap='Greens', subset=['number_of_suicide'])


# Suicide number with age groups

# ###### The highest number of suicides was in 1999. 
# ###### The lowest number of suicides was in 2016.

# ##### Suicide number with age and sex

# In[17]:


import matplotlib.pyplot as plt
plt = plt.subplots(1, 1, figsize = (16 ,6))
plt1 = sns.barplot(x = df['age_group'], y = 'number_of_suicide', hue='sex', data = df, palette = 'Accent')


# Males are, in all the years, more likely to commit suicide than females. Early edulthood and adulthood are the age groups with the most suicide rate. 

# ##### Suicide number with generation

# In[18]:


import matplotlib.pyplot as plt
# suicide based on generation 
df.generation.dropna(inplace = True)
labels = df.generation.value_counts().index
sizes = df.generation.value_counts().values
plt.figure(0,figsize = (7,7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Suicide According to Generation',color = 'black',fontsize = 17)
plt.show()


# ##### Suicide number with generation and sex

# In[19]:


import matplotlib.pyplot as plt
plt = plt.subplots(1, 1, figsize = (16 ,6))
plt2 = sns.barplot(x = df['generation'], y = 'number_of_suicide',
                  hue='sex',data=df, palette='autumn')


# This visualizations shows that generation boomers have the highest suicide rate.

# ##### Number of suicide by country

# In[20]:


# top 10 countries by most suicide number
import matplotlib.pyplot as plt
plt = plt.subplots(1, 1, figsize = (16 ,6))
dfcountry = df['number_of_suicide'].groupby(df.country).sum().sort_values(ascending=False)
plt3 = sns.barplot(dfcountry.head(10), dfcountry.head(10).index, palette='Reds_r')


# In[21]:


# least 10 countries by most suicide number
import matplotlib.pyplot as plt
plt = plt.subplots(1, 1, figsize = (16 ,6))
dfcountry = df['number_of_suicide'].groupby(df.country).sum().sort_values(ascending=False)
plt4 = sns.barplot(dfcountry.tail(10), dfcountry.tail(10).index, palette='Blues_r')


# Russian Federation leads the world by number of suicide and Dominica records the least nu,ber of suicides in the world from 1968 to 2016

# In[22]:


# To have a more accurate representation of suicide by country let us compare by their suicide number per 100 thousand
# top 10 countries by most suicide number per 100 thousand people
import matplotlib.pyplot as plt
plt = plt.subplots(1, 1, figsize = (16 ,6))
dfcountry = df['suicide_per_100_thousand'].groupby(df.country).sum().sort_values(ascending=False)
plt4 = sns.barplot(dfcountry.head(10), dfcountry.head(10).index, palette='Reds_r')


# In[23]:


# least 10 countries by most suicide number per 100 thousand people
import matplotlib.pyplot as plt
plt = plt.subplots(1, 1, figsize = (16 ,6))
dfcountry = df['suicide_per_100_thousand'].groupby(df.country).sum().sort_values(ascending=False)
plt4 = sns.barplot(dfcountry.tail(10), dfcountry.tail(10).index, palette='Blues_r')


# The Russian Federation still leads and Dominica is the last; however, we can see some variations in other countries. We can also see how close mose countries are when compared using suicide per 100 thousand. 

# Number of suicide and GDP

# In[24]:


(df.dtypes=="object").index[df.dtypes=="object"]


# In[25]:


# number of suicides and GDP per capita 
import matplotlib.pyplot as plt
f, ax = plt.subplots(1,1, figsize=(10,8))
plt5 = sns.scatterplot(x="gdp_per_capita", y="number_of_suicide", data=df, color='purple')


# Prediction 

# In[26]:


df.corr()


# In[27]:


# plot a heat map for the correlation
import matplotlib.pyplot as plt
plt = plt.subplots(1, 1, figsize = (16 ,6))
plt6 = sns.heatmap(df.corr(),annot=True, cmap='coolwarm')


# 

# In[28]:


from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor 
import sklearn.metrics
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(df[['population', 'sex', 'hdi', 'year', 'gdp_per_capita', 'age_group']],
                                                    df['number_of_suicide'],test_size=.4, random_state=42)


# In[29]:


y = df.number_of_suicide
X = df[['population', 'sex', 'hdi', 'age_group']]
X = sm.add_constant(X)  # Adds a constant term to the predictor

# Build the regression with OLS
lr_model = sm.OLS(y, X).fit()
print(lr_model.summary())


# In[30]:


linearM = LinearRegression()
linearM.fit(X_train, y_train)


# In[31]:


y_pred = linearM.predict(X_train)
y_pred


# In[32]:


mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
rmse


# In[33]:


sklearn.metrics.r2_score(y_train, y_pred)


# In[34]:


RandomForest = RandomForestRegressor()
RandomForest.fit(X_train, y_train)


# In[35]:


y_predi = RandomForest.predict(X_train)
y_predi


# In[36]:


mse = mean_squared_error(y_train, y_predi)
rmse = np.sqrt(mse)
rmse


# In[37]:


sklearn.metrics.r2_score(y_train, y_predi)


# In[38]:


voting = VotingRegressor([('lr', linearM), ('rf', RandomForest)])
y_prediction = voting.fit(X_train, y_train).predict(X_train)
y_prediction


# In[39]:


mse = mean_squared_error(y_train, y_prediction)
rmse = np.sqrt(mse)
rmse


# In[40]:


sklearn.metrics.r2_score(y_train, y_prediction)


# In[41]:


y_pred = linearM.predict(X_test)
y_pred


# In[42]:


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse


# In[43]:


sklearn.metrics.r2_score(y_test, y_pred)


# In[44]:


y_predi = RandomForest.predict(X_test)
y_predi


# In[45]:


mse = mean_squared_error(y_test, y_predi)
rmse = np.sqrt(mse)
rmse


# In[46]:


sklearn.metrics.r2_score(y_test, y_predi)


# In[47]:


y_prediction = voting.predict(X_test)
y_prediction


# In[48]:


mse = mean_squared_error(y_test, y_prediction)
rmse = np.sqrt(mse)
rmse


# In[49]:


sklearn.metrics.r2_score(y_test, y_prediction)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




