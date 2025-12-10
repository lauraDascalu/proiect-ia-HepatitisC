import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

df = pd.read_csv("source/HepatitisCdata.csv", index_col=0)
# print(df.dtypes)

print(f"Unique values from target variable Category: {df.Category.unique()}")

#handling categorical values
df['Category'] = df['Category'].map({'0=Blood Donor': 0, '0s=suspect Blood Donor': 0, "1=Hepatitis" : 1, "2=Fibrosis" : 1, "3=Cirrhosis" : 1})
#print(df.head())
df = pd.get_dummies(df, columns=['Sex'], drop_first=True, prefix='Sex')
df = df.rename(columns={'Sex_m': 'Sex_Male'})
print(df.head())
print(df.dtypes)

#handling missing values with median imputation
print(f"Missing values count: {df.isna().sum()}")
df.fillna(df.median(), inplace=True)
print(f"Missing values count: {df.isna().sum()}")

#correlation matrix
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt='.1g', cmap="Blues_r", cbar=False, linewidths=0.5, linecolor='grey');
plt.title('Correlation Matrix')
plt.show()