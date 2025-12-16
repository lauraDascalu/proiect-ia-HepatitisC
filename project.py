import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("source/HepatitisCdata.csv", index_col=0)
# print(df.dtypes)

print(f"Unique values from target variable Category: {df.Category.unique()}")

#handling categorical values
df['Category'] = df['Category'].map({'0=Blood Donor': -1, '0s=suspect Blood Donor': -1, "1=Hepatitis" : 1, "2=Fibrosis" : 1, "3=Cirrhosis" : 1})
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
sns.heatmap(df.corr(), annot=True, fmt='.1f', cmap="Blues_r", cbar=False, linewidths=0.5, linecolor='grey');
# plt.title('Correlation Matrix')
#plt.show()

correlation_matrix = df.corr()
target_correlations = correlation_matrix['Category'].drop('Category')
threshold_strong = 0.4
threshold_weak = 0.2
strong_corr_vars = target_correlations[abs(target_correlations) >= threshold_strong].sort_values(ascending=False)
weak_corr_vars = target_correlations[abs(target_correlations) < threshold_weak].sort_values(ascending=False)
print("Strongly correlated features with 'Category':")
print(strong_corr_vars.to_string())
print("Weakly correlated features with 'Category':")
print(weak_corr_vars.to_string())

df_features = df.drop(columns=['Category'])
feature_correlation_matrix = df_features.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(feature_correlation_matrix,annot=True, fmt='.1f', cmap='viridis', cbar=False,linewidths=0.5,linecolor='grey')
# plt.title('Correlation in between features')
# plt.show()
stacked_correlations = feature_correlation_matrix.unstack().sort_values(ascending=False)
multicol_threshold=0.8
too_correlated_vars = stacked_correlations[abs(stacked_correlations) >= multicol_threshold].sort_values(ascending=False)
too_correlated_vars = too_correlated_vars[abs(too_correlated_vars) < 1.0 ]
if not too_correlated_vars.empty:
    print(f"Highly correlated features: {too_correlated_vars}")
else:
    print("None of the features are highly correlated.")

#split into traing and test dataset
X = df.drop(columns=['Category'])
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X,  y, test_size=0.2,random_state=42,)

#scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

#print(X_train_scaled.head())

df_train_scaled = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
df_train_scaled.to_csv("source/train_data.csv", index=False)

# df_test_scaled = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)

# print(df_train_scaled.Category.unique())
# print(df_test_scaled.Category.unique())

X_test_scaled.to_csv("source/test_data.csv", index=False)
y_test.to_csv("source/test_y.csv", index=False)