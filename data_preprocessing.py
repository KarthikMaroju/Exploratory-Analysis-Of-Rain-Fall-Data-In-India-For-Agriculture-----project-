# Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import Dataset
df = pd.read_csv("dataset/rainfall.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Handling Missing Values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Data Visualization
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Splitting Dependent & Independent Variables
X = df.drop("Rainfall", axis=1)
y = df["Rainfall"]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data Preprocessing Completed Successfully")
