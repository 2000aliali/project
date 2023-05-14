#!/usr/bin/env python
# coding: utf-8

# # Projet : Modèle de Machine Learning pour prédier le score de qualité d' un vin

# ##### imopretation des bibliotique 

# In[165]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR  
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[ ]:





# In[166]:


df1=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",sep=";")
df2=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",sep=";")
df1["type_vin"]=0
df2["type_vin"]=1
df=pd.concat([df1, df2])
df.head(20)


# In[167]:


df.info()


# In[168]:


df["quality"].value_counts(normalize=True)


# # Analyse exploratoire

# In[169]:


df.describe()


# In[170]:


# pas de donner manquante 
df.isna().sum()


# #### supposon que un vin qui est une qality> 7 est un bon vin et vire se vire ça 

# In[171]:


df3=df[df["quality"]>7]
df3.head()
df3.info()


# In[ ]:





# In[172]:


df3["quality"].value_counts(normalize=True)


# In[173]:


df3.describe()


# In[174]:


df4=df[df["quality"]<=7]
#df4.head()
#df4.info()


# In[175]:


df4.describe()


# In[176]:


df.hist(figsize=(14,6));


# In[177]:


df.corr()


# In[178]:


sns.heatmap(df.corr())


# ##### prétraitement 

# In[179]:


x=df.drop("quality",axis=1)
y=df["quality"]
seed=33
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=seed,test_size=0.2)
print("x_train.shape=",x_train.shape)
print("y_train.shape=",y_train.shape)
print("x_test.shape=",x_test.shape)
print("y_test.shape=",y_test.shape)


# In[180]:


scaler=StandardScaler()
x_train_scaller=scaler.fit_transform(x_train)
x_test_scaller=scaler.fit_transform(x_test)


# In[181]:


x_train_scaller


# In[182]:


'''lr_cve=cross_val_score(
    linearRegession(),
    x_train_scalle,
    y_train,
    cv=3,
    scoring="neg_root_mean_squarederor"
)
rf_cve=cross_val_score(
    RandomForestRegressor(random_state=seed),
    x_train_scalle,
    y_train,
    cv=3,
    scoring="neg_root_mean_squarederor"
) 
svm_cve=cross_val_score(
    SVR(),
    x_train_scalle,
    y_train,
    cv=3,
    scoring="neg_root_mean_squarederor"
)'''


lr_cve = cross_val_score(
    LinearRegression(),
    x_train_scaller,
    y_train,
    cv=3,
    scoring="neg_root_mean_squared_error"
)
rf_cve = cross_val_score(
    RandomForestRegressor(random_state=seed),
    x_train_scaller,
    y_train,
    cv=3,
    scoring="neg_root_mean_squared_error"
)
svm_cve = cross_val_score(
    SVR(),
    x_train_scaller,
    y_train,
    cv=3,
    scoring="neg_root_mean_squared_error"
)

    


# In[183]:


lr_cve


# In[184]:


svm_cve


# In[185]:


rf_cve


# In[186]:


print("RF :",rf_cve.mean(),"LR :",lr_cve.mean(),"SVM :",svm_cve.mean())


# ##### Construction du modéle

# In[187]:


model=RandomForestRegressor(random_state=seed)
model.fit(x_train_scaller,y_train)


# In[188]:


y_pred=model.predict(x_test_scaller)


# In[189]:


y_pred


# In[190]:


mean_squared_error(y_test,y_pred,squared=False)


# In[191]:


7+0.63


# In[192]:


7-0.63


# ##### Conclusion Attribus importants

# In[193]:


model.feature_importances_


# In[194]:


vars_imp=pd.Series(model.feature_importances_,index=x.columns).sort_values(ascending=0)
vars_imp             


# ###### donc pour produier des vines de bonnne qualié il faut agier ces les premier 4 parameter quellque soit le type de vin :
# alcohol                 
# volatile acidity        
# free sulfur dioxide     
# sulphates  
# 

# In[195]:


plt.figure(figsize=(12,10))
sns.barplot(x=vars_imp,y=vars_imp.index);
plt.title("importance de chaque predecteur")
plt.show()


# In[ ]:




