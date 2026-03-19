# importing liabraries
# import warnings
# warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# importing data 

df = pd.read_csv("insurance.csv")
print(df.head()) 

# Prforming EDA
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
    plt.show()


