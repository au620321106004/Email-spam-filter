#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # Data Pre Processing

# In[2]:


#Loading The Dataset
df = pd.read_csv(r"D:\New folder\Email Spam Filter Data Sets\spam.csv",encoding='latin-1')
df


# # Description

# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


#Replacing Null ("NaN")values with Null string
df.isnull().sum()


# In[6]:


df.isnull().any()


# In[7]:


# Checking the number of rows and columns
df.shape


# In[8]:


df.dtypes


# In[9]:


#Removing NaN
df1 = df.where(pd.notnull(df),'')
df1


# In[10]:



df1.columns


# In[11]:


#Renaming columns
df1.rename(columns={'v1':'Category','v2':'Messages'},inplace=True)
#Dropping Empty Columns
df1.drop(columns=['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'],axis=1)


# In[12]:


#Label Encoding spam & ham
df1.loc[df1['Category'] == 'spam', 'Category'] = 0
df1.loc[df1['Category'] == 'ham', 'Category'] = 1


# In[13]:


# Seperating the text as texts and label
X = df1['Messages']
Y = df1['Category']


# In[14]:


X.head()


# In[15]:


Y.head()


# In[16]:


from sklearn.model_selection import train_test_split
X_Train,X_test,Y_Train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)


# In[17]:


X.shape


# In[18]:


Y.shape


# In[19]:


X_Train.shape


# In[20]:


Y_test.shape


# # Feature Extraction

# In[21]:


# Transform text data to feature vectors that can be used as input to the logistic regression
from sklearn.feature_extraction.text import TfidfVectorizer
feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase='True')


# In[22]:


X_train_feature = feature_extraction.fit_transform(X_Train)
X_test_feature = feature_extraction.transform(X_test)


# In[23]:


#  Converting Y_train and T_test as Int(Integers)
Y_Train = Y_Train.astype('int')
Y_test = Y_test.astype('int')


# In[24]:


print(X_train_feature)


# In[25]:


X_Train


# # Training the Model

# In[26]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[27]:


Y_Train


# In[28]:


model.fit(X_train_feature,Y_Train)


# In[29]:


#Evaluation & Prediction of Training Model


# In[30]:


from sklearn.metrics import accuracy_score


# In[31]:


y_pred1 = model.predict(X_train_feature)
acc1 = accuracy_score(Y_Train,y_pred1)


# In[32]:


print("Training Accuracy is : ",acc1 * 100)


# In[33]:


#Prediciton of Test data
y_pred2 = model.predict(X_test_feature)
acc2 = accuracy_score(Y_test,y_pred2)


# In[34]:


print("Training Accuracy is : ",acc2 * 100)


# In[35]:


#  Building a Predictable System
input_mail = ["As a valued customer, I am pleased to advise you that following recent review of your Mob No. you are awarded with a £1500 Bonus Prize, call 09066364589"]

# Convert Text to feature vectors
input_data_feature = feature_extraction.transform(input_mail)

# Making Prediction
prediction = model.predict(input_data_feature)

print(prediction)

if(prediction == [1]):
    print("This is the Ham Mail.")
else:
    print("This is the Spam Mail.")


# In[36]:


#  Building a Predictable System
input_mail = ["Hello,How are You?"]

# Convert Text to feature vectors
input_data_feature = feature_extraction.transform(input_mail)

# Making Prediction
prediction = model.predict(input_data_feature)

print(prediction)

if(prediction == [1]):
    print("This is the Ham Mail.")
else:
    print("This is the Spam Mail.")


# In[ ]:



