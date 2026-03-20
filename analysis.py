# importing liabraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# importing data 

df = pd.read_csv("insurance.csv")
print(df.head()) 

# Performing EDA
print(f'\n shape of the data frame is {df.shape}\n')
print(df.info())
print(f'\n decriptiion of the given dataset\n {df.describe()}\n')
# now check is there any null value(removing null value is part of data cleaning)
print(df.isnull().sum())
print(f'\n columns in given dataset:\n {df.columns}\n')
# extracting numeric columns
numeric_col=['age', 'bmi', 'children', 'charges']
for col in numeric_col:
    plt.figure(figsize=(8,6))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
    # plt.savefig(f"Distribution of {col}")
    plt.show()

# getting countplot
sns.countplot(x=df['smoker'])
plt.show()
# creating boxplots
for col in numeric_col:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.savefig(f"{col}")
    plt.show()
# plotting heatmap for correlation
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True),annot=True)
# plt.savefig("correlation")
plt.show()

#now data cleaning and processing
# 1.lets take the data by new name i.e df_cleaned
df_cleaned=df.copy()
print(df_cleaned.head())
print(df_cleaned.shape)

# now drop duplicates value in the data

df_cleaned.drop_duplicates(inplace=True)
print(df_cleaned.isnull().sum())
# checking data types of the given data so that we can change the object data types
print(df_cleaned.dtypes)
# now conversion of object datatypes
print(df_cleaned['sex'].value_counts())
# we can number of males and females
# now conversion starts(males into 0 and females into 1)
df_cleaned['sex']=df_cleaned['sex'].map({"male":0,"female":1})
print(df_cleaned.head())
# now same for smoker
print(df_cleaned["smoker"].value_counts())
# we have no and yes so now we will start converting them into 0 and 1
df_cleaned["smoker"]=df_cleaned["smoker"].map({"no":0,"yes":1})
print(df_cleaned.head())
# now rename above changed values
df_cleaned.rename(columns={
    "sex":"is_female",
    "smoker":"is_smoker"
},inplace=True)
print(df_cleaned.head())
# now changing for region counts
print(df['region'].value_counts())
# now do hotcoding for this, to simplify the same by making the different columns name like northeast and southeast
# for this we have to call function get dummies adn we have to make DropFirst=true because otherwise it would conside region columns as dummy column

df_cleaned=pd.get_dummies(df_cleaned,columns=["region"],drop_first=True)
print(df_cleaned.head())
# now changing true false value to 1 and 0
df_cleaned=df_cleaned.astype(int)
print(df_cleaned.head())

# Feature Engineering and Extraction
# it is required to delete and add columns that will be necessary for the machine learning models
sns.histplot(df["bmi"])
# plt.savefig("bmi")
plt.show()
# 1.Now create bmi_category
df_cleaned["bmi_category"]=pd.cut(
    df_cleaned["bmi"],
    bins=[0,18.5,24.9,29.9,float('inf')],
    labels=["underweight","normal","overweight","obese"]

)
print(df_cleaned.head())
# now do hotcoding for making it inot integers
df_cleaned=pd.get_dummies(df_cleaned,columns=["bmi_category"],drop_first=True)
df_cleaned=df_cleaned.astype(int)
print(df_cleaned.head())


