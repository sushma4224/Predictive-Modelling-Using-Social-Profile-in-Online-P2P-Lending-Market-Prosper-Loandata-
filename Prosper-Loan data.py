#!/usr/bin/env python
# coding: utf-8

# #   Predictive Modelling Using Social Profile in Online P2P Lending Market

# # Prosper Loandata
# 

# # Problem Statement :

# Online peer-to-peer (P2P) lending markets enable individual consumers to borrow from, and lend money to, one another directly. We study the borrower-,loan- and group- related determinants of performance predictability in an online P2P lending market by conceptualizing financial and social strength to predict borrower rate and whether the loan would be timely paid.

# Importing all necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings ('ignore')


# Loading the Data set: We have 113937 rows × 81 columns

# In[2]:


df= pd.read_csv("C:/Users/murthy/Downloads/prosperLoanData.csv")


# In[3]:


df


# In[5]:


df.head()


# Statistical Information

# In[4]:


df.shape


# In[6]:


df.dtypes


# In[7]:


df.describe()


# Checking Data type of Attributes

# In[8]:


df.info()


# Checking Unique values in dataset :

# In[9]:


df.apply(lambda x: len(x.unique()))


# # Preprocessing the Dataset

# Checking duplicate values:

# In[35]:


df.duplicated().sum()


# We observe that dataset contains no duplicate values.

# Checking Missing (null) values:

# In[10]:


df.isnull().sum()


# We observe many attributes have missing values.

# In[11]:


print(df.isnull().values.sum())


# We have Categorical as well as numerical attributes which we will process seperately.

# In[17]:


categorical=df.select_dtypes("object")
categorical


# In[18]:


continuous=df.select_dtypes("number")
continuous


# Checking Missing (null) values of Categorical attributes:

# In[37]:


categorical.isna().sum()


# Filling Missing values of categorical attributes using Mode funcion:

# In[20]:


categorical=categorical.fillna(categorical.mode().iloc[0])
categorical


# In[21]:


categorical.isna().sum()


# All the missing values of categorical attributes are now filled.

# Checking Missing (null) values of Numerical attributes:

# In[39]:


continuous.isna().sum()


# Filling Missing values of Numerical attributes using Median funcion:

# In[22]:


continuous=continuous.fillna(continuous.median().iloc[0])
continuous


# In[23]:


continuous.isna().sum()


# All the missing values of Numerical attributes are now filled.

# Combining the Categorical continuous data into one file named "ndf" using concat function with the inner join condition.

# In[24]:


ndf=pd.concat([categorical,continuous],axis=1,join='inner')
ndf


# Checking again wether there is still have missing values or not

# In[25]:


ndf.isna().sum()


# Displaying column names

# In[26]:


ndf.columns


# In[27]:


#ndf['LoanOriginationDate']=pd.to_datetime(ndf['LoanOriginationDate'])


# # Data Cleaning:

# Removing unnecessary columns from the dataset

# In[28]:


ndf.drop(['ListingCreationDate','LoanOriginationDate','GroupKey','CreditGrade','ProsperPrincipalBorrowed','ProsperPrincipalOutstanding','EstimatedEffectiveYield','EstimatedLoss','EstimatedReturn','ProsperRating (numeric)','TotalProsperLoans','TotalProsperPaymentsBilled','OnTimeProsperPayments','ProsperPaymentsLessThanOneMonthLate','ProsperPaymentsOneMonthPlusLate','ListingKey'],axis=1,inplace=True)


# In[29]:


ndf.head()


# # Label Encoding:

# Label Encoding is to convert the categorical column into the numerical column.

# In[30]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cols=['ClosedDate','ProsperRating (Alpha)','BorrowerState','Occupation','EmploymentStatus','FirstRecordedCreditLine','DateCreditPulled','IncomeRange','LoanKey','LoanOriginationQuarter','MemberKey']
ndf[cols]=ndf[cols].apply(LabelEncoder().fit_transform)
ndf.head()


# One Hot Encoding:

# we can also use one hot encoding for categorical columns.

# In[31]:


Ls=pd.get_dummies(ndf['LoanStatus'])
print(Ls)


# It will create a new column for each category. Hence, it will add the corresponding category instead of numerical values. if the corresponding location type i present it will show as "1" orelse it will show "0"

# In[32]:


Ndf=pd.concat((Ls,ndf),axis=1)
Ndf=Ndf.drop(['LoanStatus'],axis=1)
Ndf=Ndf.drop(['Cancelled','Chargedoff','Current','Defaulted','FinalPaymentInProgress','FinalPaymentInProgress','Past Due (1-15 days)','Past Due (16-30 days)','Past Due (31-60 days)','Past Due (61-90 days)','Past Due (91-120 days)','Past Due (>120 days)'],axis=1)
Newdf=Ndf.rename(columns={"Completed":"LoanStatus"})
print(Newdf)


# Here Loan Status 1 = Completed and 0 = Not Completed

# In[33]:


Newdf


# In[34]:


Newdf.columns


# In[ ]:




