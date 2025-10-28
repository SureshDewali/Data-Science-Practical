# Practical 3: Data Cleaning and Preprocessing with Pandas

import pandas as pd

print("Name: Suresh Dewali")
print("Roll No: 1323575")

# 1️⃣ Create a small sample dataset (you can also load from CSV)
data = pd.DataFrame({
    'Name': ['Ram', 'Sita', 'Hari', 'Gita', 'Ram', 'Mina', 'Kiran', None],
    'Age': [25, 28, None, 32, 25, 29, None, 24],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', None, 'Male', 'Female'],
    'Salary': [30000, 35000, 40000, 42000, 30000, 38000, 36000, 31000]
})

print("Roll no:1323575")
print("Name:Suresh Dewali")
print("\nInitial Data:\n", data)

# 2️⃣ Check for missing values
print("\nMissing Values:\n", data.isnull().sum())

# 🔹 Fill missing values (modern syntax – no warnings)
data = data.fillna({
    'Age': data['Age'].mean(),
    'Gender': data['Gender'].mode()[0],
    'Name': 'Unknown'
})

# 3️⃣ Remove duplicates
data = data.drop_duplicates()
print("\nAfter Removing Duplicates:\n", data)

# 4️⃣ Use groupby(), describe(), and filtering
print("\nAverage Age by Gender:\n", data.groupby('Gender')['Age'].mean())
print("\nDescriptive Statistics:\n", data.describe())

# 5️⃣ Filtering example – select records where Age > 30
filtered_data = data[data['Age'] > 30]
print("\nFiltered Data (Age > 30):\n", filtered_data)

# 6️⃣ Save cleaned data (optional)
data.to_csv("cleaned_data.csv", index=False)
print("\n✅ Cleaned data saved as 'cleaned_data.csv'")
