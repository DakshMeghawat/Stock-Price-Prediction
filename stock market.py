#!/usr/bin/env python
# coding: utf-8

# In[1]:


print ("hello")


# In[8]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn


# In[9]:


import yfinance as yf


# In[10]:


sp500=yf.Ticker("^GSPC")


# In[12]:


sp500 = sp500.history(period="max")


# In[13]:


sp500


# In[14]:


sp500.index


# In[75]:


sp500.plot.line(y= "Close", use_index=True)


# In[15]:


del sp500["Dividends"]
del sp500["Stock Splits"]


# In[16]:


sp500["Tomorrow"]=sp500["Close"].shift(-1)


# In[78]:


sp500


# In[18]:


sp500["Target"]= (sp500["Tomorrow"]>sp500["Close"]).astype(int)


# In[19]:


sp500


# In[20]:


sp500 = sp500.loc["1990-01-01":].copy()


# In[21]:


sp500


# In[22]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, min_samples_split=100,random_state=1)

train = sp500.iloc[:-100]
test=sp500.iloc[-100:]

predictors= ["Close", "Volume" , "Open" , "High" , "Low"]
model.fit(train[predictors], train["Target"])


# In[23]:


from sklearn.metrics import precision_score

preds = model.predict(test[predictors])


# In[24]:


import pandas as pd

preds = pd.Series(preds, index=test.index)


# In[25]:


precision_score(test["Target"], preds)


# In[26]:


combined = pd.concat([test["Target"], preds], axis=1)


# In[27]:


combined.plot()


# In[49]:


def predict(train , test ,predictors , model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index,name="Predictions")
    combined = pd.concat([test["Target"],preds],axis=1)
    return combined


# In[47]:


def backtest (data, model, predictors , start =2500, step=250):
    all_predictions =[]
    
    for i in range (start, data.shape[0] , step):
        train =data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions =predict(train , test ,predictors , model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


# In[48]:


predictions= backtest(sp500, model, predictors)


# In[44]:


predictions["Predictions"].value_counts()


# In[45]:


precision_score(predictions["Target"] ,predictions["Predictions"])


# In[58]:


horizons = [2,5,60,250,1000]
new_predictors =[]
for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    ration_column =f"Close_Rario_{horizon}"
    sp500[ratio_column]=sp500["close"] / rolling_averages["Close"]
    trend_column = f"Trend_{horizon}"
    sp500[trend_column]=sp500.shift(1).rolling(horizon).sum()["Target"]
        
        
    new_predictors+= [ratio_column,trend_column]    
        
        
        


# In[ ]:





# In[ ]:




