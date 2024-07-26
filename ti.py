import pandas as pd
import numpy as np
import streamlit as st


# In[53]:


df=pd.read_csv('swiggy_cleaned.csv')





df['time_minutes']=df['time_minutes'].str.split('-').str[0]


# In[57]:


df['time_minutes']=df['time_minutes'].astype(float)


# In[58]:


df['time_minutes']=df['time_minutes'].fillna(df['time_minutes'].mean())







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





# In[176]:


n=pipe.predict(pd.DataFrame(columns=['hotel_name', 'rating','food_type', 'location','offer_above', 'offer_percentage'],data=np.array(['KFC',4.2,'Burgers, Biryani, American, Snacks, Fast Food','Kandivali East',80,40]).reshape(1,6)))


# In[177]:


#df[df['hotel_name']=='KFC']


# In[178]:


from sklearn.metrics import r2_score


# In[179]:


#r2_score(y_test, pipe.predict(X_test))
st.header('Delivery Time Predication APP')
part=st.sidebar.radio(" ", ["Prediction", "Analysis"])
if part=='Prediction':
    col1,col2,col3=st.columns(3)
    with col1:
        a=st.selectbox('Hotel Name',df['hotel_name'].unique())
    with col2:
        b=st.number_input('Rating')
    with col3:
        c=st.selectbox('Food Type',df['food_type'].unique())
    col1,col2,col3=st.columns(3)
    with col1:
        d=st.selectbox('Location',df['location'].unique())
    with col2:
        e=st.number_input('offer_above')
    with col3:
        f=st.number_input('offer_percentage')
    
    n=pipe.predict(pd.DataFrame(columns=['hotel_name', 'rating','food_type', 'location','offer_above', 'offer_percentage'],data=np.array([a,b,c,d,e,f]).reshape(1,6)))
    if st.button('Predict Time'):
        st.write('Delivery Time'+str(int(n[0])))

else:
    import plotly.express as px
    col1,col2,col3=st.columns(3)
    with col1:
        r=st.slider('Budget',min_value=100, max_value=10000, value=100, step=100)
    with col2:
        n=st.selectbox('Location',df['location'].unique())
    with col3:
        o=st.slider('% OFF You Need',min_value=0, max_value=100, value=10, step=10)
    x=df[(df['offer_above']<=r)&(df['offer_percentage']>=o)&(df['location']==n)].groupby('hotel_name')['time_minutes'].mean().nlargest(10)
    if x.shape[0]!=0:
        fig1=px.line(x)\
        fig1.update_layout(
            title={
                'text': "Hotel Vs Time Taken",
                'y':0.9,
                'x':0.9,
                'xanchor': 'right',
                'yanchor': 'top'})
        fig1.update_layout(
            title="Hotel Vs Time Taken",
            xaxis_title="Hotel Name",
            yaxis_title="Time Of Delivery",
        )
        st.write(fig1)
    y=df[(df['offer_above']<=r)&(df['offer_percentage']>=o)&(df['location']==n)].groupby('hotel_name')['rating'].mean().nlargest(10)
    if y.shape[0]!=0:  
        fig2=px.line(y)
        fig2.update_layout(
            title={
                'text': "Hotel Vs Rating",
                'y':0.9,
                'x':0.9,
                'xanchor': 'right',
                'yanchor': 'top'})
        fig2.update_layout(
            title="Hotel Vs Rating",
            xaxis_title="Hotel Name",
            yaxis_title="Rating",
            legend_title="Legend Title",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
            )
        st.write(fig1)
    
    



