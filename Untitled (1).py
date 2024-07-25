#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np


# In[53]:


df=pd.read_csv('swiggy_cleaned.csv')


# In[54]:


df


# In[55]:


df.info()


# In[56]:


df['time_minutes']=df['time_minutes'].str.split('-').str[0]


# In[57]:


df['time_minutes']=df['time_minutes'].astype(float)


# In[58]:


df['time_minutes']=df['time_minutes'].fillna(df['time_minutes'].mean())


# In[59]:


df.info()


# In[60]:


df['offer_above'].unique()


# In[61]:


df['offer_percentage']=df['offer_percentage'].str.replace('not_available','0')


# In[62]:


df['offer_percentage']=df['offer_percentage'].astype(int)


# In[63]:


df['offer_above']=df['offer_above'].str.replace('not_available','10000')
df['offer_above']=df['offer_above'].str.replace('FREE ITEM','0')


# In[64]:


df['offer_above']=df['offer_above'].str.replace('20% OFF','100000')


# In[65]:


df['offer_above']=df['offer_above'].astype(int)


# In[66]:


df=df[df['offer_above']!=100000]


# In[67]:


df


# In[68]:


df['rating']=df['rating'].str.split(' ').str[0]


# In[69]:


df['rating']=df['rating'].astype(float)


# In[70]:


df=df[df['rating']<=5]


# In[71]:


x=df.drop(['time_minutes'],axis=1)


# In[72]:


y=df['time_minutes']


# In[73]:


from sklearn.model_selection import train_test_split


# In[74]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)


# In[75]:


X_train


# In[76]:


from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()


# In[77]:


from sklearn.compose import ColumnTransformer


# In[78]:


ohe.fit([['hotel_name','food_type','location']])
trf=ColumnTransformer([
    ('trf',OneHotEncoder(max_categories=6,sparse_output=False,handle_unknown = 'ignore'),['hotel_name','food_type','location'])]
,remainder='passthrough')


# In[79]:


ohe.categories_


# In[109]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from  sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


# In[173]:


pipe=Pipeline(steps=[
    ('step1',trf),
    ('step2', KNeighborsRegressor(n_neighbors=9))
]
)


# In[174]:


pipe.fit(X_train,y_train)


# In[175]:


df.columns


# In[176]:


pipe.predict(pd.DataFrame(columns=['hotel_name', 'rating','food_type', 'location','offer_above', 'offer_percentage'],data=np.array(['KFC',4.2,'Burgers, Biryani, American, Snacks, Fast Food','Kandivali East',80,40]).reshape(1,6)))


# In[177]:


df[df['hotel_name']=='KFC']


# In[178]:


from sklearn.metrics import r2_score


# In[179]:


r2_score(y_test, pipe.predict(X_test))


# In[ ]:





# In[ ]:




