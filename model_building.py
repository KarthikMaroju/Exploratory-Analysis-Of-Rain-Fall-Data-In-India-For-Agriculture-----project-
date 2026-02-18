import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("dataset/rainfall.csv")

df.fillna(df.mean(numeric_only=True), inplace=True)

X = df.drop("Rainfall", axis=1)
y = df["Rainfall"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initializing Model
model = LinearRegression()

# Training Model
model.fit(X_train, y_train)

# Evaluation
score = model.score(X_test, y_test)
print("Model Accuracy (R2 Score):", score)

# Save Model
pickle.dump(model, open("model/rainfall_model.pkl", "wb"))

print("Model Saved Successfully")

