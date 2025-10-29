import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

print("Name: Suresh Dewali")
print("Roll No: 1323575\n")

# -------------------------------
# Step 1: Create a Sample Dataset
# -------------------------------
data = pd.DataFrame({
    'Name': ['Ram', 'Sita', 'Hari', 'Gita', 'Kiran', 'Mina', 'John', 'Ravi'],
    'Age': [25, 28, np.nan, 32, 29, np.nan, 35, 40],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male'],
    'Department': ['HR', 'IT', 'Finance', 'IT', 'Finance', 'HR', 'IT', 'Finance'],
    'Salary': [30000, 48000, 40000, np.nan, 52000, 39000, 45000, 61000]
})

print("📘 Original Dataset:\n", data, "\n")

# ------------------------------------
# Step 2: Handle Missing Data
# ------------------------------------
imputer = SimpleImputer(strategy='mean')
data[['Age', 'Salary']] = imputer.fit_transform(data[['Age', 'Salary']])

print("✅ After Handling Missing Data:\n", data, "\n")

# ------------------------------------
# Step 3: Encode Categorical Variables
# ------------------------------------
# Encode Gender (Label Encoding)
label_encoder = LabelEncoder()
data['Gender_Encoded'] = label_encoder.fit_transform(data['Gender'])

# One-Hot Encode Department
dept_encoded = pd.get_dummies(data['Department'], prefix='Dept')
data = pd.concat([data, dept_encoded], axis=1)

print("✅ After Encoding Categorical Variables:\n", data, "\n")

# ------------------------------------
# Step 4: Feature Scaling
# ------------------------------------
scaler_std = StandardScaler()
scaler_minmax = MinMaxScaler()

data['Age_std'] = scaler_std.fit_transform(data[['Age']])
data['Salary_std'] = scaler_std.fit_transform(data[['Salary']])
data['Age_minmax'] = scaler_minmax.fit_transform(data[['Age']])
data['Salary_minmax'] = scaler_minmax.fit_transform(data[['Salary']])

print("✅ After Feature Scaling:\n", data, "\n")

# ------------------------------------
# Step 5: Split Dataset into Train/Test
# ------------------------------------
X = data[['Age', 'Salary', 'Gender_Encoded', 'Dept_Finance', 'Dept_HR', 'Dept_IT']]
y = data['Salary']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("✅ Training Features (X_train):\n", X_train, "\n")
print("✅ Testing Features (X_test):\n", X_test, "\n")
print("✅ Training Labels (y_train):\n", y_train, "\n")
print("✅ Testing Labels (y_test):\n", y_test, "\n")

print("🎯 Outcome: Data is preprocessed, encoded, scaled, and split successfully for ML models.")
