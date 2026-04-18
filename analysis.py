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
# creating boxplots(for matching input and output)
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

# now conversion of object datatypes to numeric data types

print(df_cleaned['sex'].value_counts())
# we can count number of males and females
# now conversion starts(males into 0 and females into 1) using map method
# it is called "labell encoding"
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
# now do "1 hotcoding" for this, to simplify the same by making the different columns name like northeast and southeast
# for this we have to call function get dummies adn we have to make DropFirst=true because otherwise it would conside region columns as dummy column

df_cleaned=pd.get_dummies(df_cleaned,columns=["region"],drop_first=True)
print(df_cleaned.head())
# now we got region into northeast , southeast etc.
# now changing true false value to 1 and 0
df_cleaned=df_cleaned.astype(int)
print(df_cleaned.head())

# Feature Engineering and Extraction

# it is required to delete and add columns that will be necessary for the machine learning models
sns.histplot(df["bmi"])
plt.savefig("bmi")
plt.show()
# we will make new columns for ml model, will divide bmi into under catergory, over weight...
# 1.Now create bmi_category
df_cleaned["bmi_category"]=pd.cut(
    df_cleaned["bmi"],
    bins=[0,18.5,24.9,29.9,float('inf')],
    labels=["underweight","normal","overweight","obese"]
)
print(df_cleaned.head())
# they are in strings so will make columns for new bmi category
# now do hotcoding for making it to integers 
df_cleaned=pd.get_dummies(df_cleaned,columns=["bmi_category"],drop_first=True)
df_cleaned=df_cleaned.astype(int)
print(df_cleaned.head())
# prepared new column i.e bmi category_normal,obese, overwight

# Now should do feature scaling of preparing machine learning model(as regression doesn't depend on value weights)
# here in this dataset only 2 column i.e Age and BMI have weighted values
print(df_cleaned.head())
from sklearn.preprocessing import StandardScaler
cols=['age','bmi','children']
# making object 
scaler=StandardScaler()
df_cleaned[cols]=scaler.fit_transform(df_cleaned[cols])
print(df_cleaned.head())

# now feature extraction...in this we select features for model
# we will take only those features which are highly co-related with charges

#  we will use pearson correlation feature to check against target
from scipy.stats import pearsonr
selected_features=['age','bmi','children','is_female','is_smoker','region_northwest','region_southeast','region_southwest','bmi_category_normal','bmi_category_overweight','bmi_category_obese']
correlations={
    feature:pearsonr(df_cleaned[feature],df_cleaned['charges'])[0]
    for feature in selected_features
}
correlation_df=pd.DataFrame(list(correlations.items()),columns=['feature','Pearsonr Correlation'])
# for making it in descend order, so we can get highest correlation value
print(correlation_df.sort_values(by='Pearsonr Correlation',ascending=False))
# by above line of code, we got the correlations
# now taking correlation for model
cat_features=['is_smoker','is_female','region_northwest','region_southeast','region_southwest','bmi_category_normal','bmi_category_overweight','bmi_category_obese']
# now importing chi square test
from scipy.stats import chi2_contingency
import pandas as pd

alpha=0.05
# to use chi square taable we have to make bins of charges so that they can corealte with the selected features i.e cat_features
from scipy.stats import chi2_contingency

alpha = 0.05

# Create bins
df_cleaned['charges_bin'] = pd.qcut(df_cleaned['charges'], q=4, labels=False)

chi2_results = {}

for col in cat_features:
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charges_bin'])
    
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)
    # above p_value will be compared with alpha 0.05
    
    decision = 'Reject Null (keep feature)' if p_val < alpha else 'Accept Null (drop feature)'
    
    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'Decision': decision
    }
chi2_df = pd.DataFrame(chi2_results).T  # <-- add .T
chi2_df = chi2_df.sort_values(by='p_value')
print(chi2_df)
# now we got which cat_features we have to keep
final_df=df_cleaned[['age','is_female','bmi','children','is_smoker','charges','region_southeast','bmi_category_obese']]
print(final_df)


# Creating model

from sklearn.model_selection import train_test_split
X=final_df.drop('charges',axis=1)
y=final_df['charges']
# splitting x and y test
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.33,random_state=42)
# in above, X_test will be used for predicting values(here we will give name y_prediction) & then
# value of prediction will be used to compare with y_test
from sklearn.linear_model import LinearRegression
model=LinearRegression()
# fitting training data in model which is Linear Regression
model.fit(X_train,y_train)
# now we have created a model linearregression which can be used for pediction 




