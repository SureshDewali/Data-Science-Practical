# Importing library
import pandas as pd

# Student Info
print("Name: Suresh Dewali")
print("Roll No: 1323575")

# 1️⃣ Create a sample Titanic-like dataset (no CSV needed)
data = {
    'Name': ['John', 'Emma', 'Robert', 'Sophia', 'Daniel', 'Olivia', 'Liam', 'Ava'],
    'Sex': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
    'Age': [22, 38, 26, 35, 30, 27, 19, 45],
    'Fare': [7.25, 71.28, 8.05, 53.10, 13.00, 26.00, 21.00, 83.50],
    'Survived': [0, 1, 0, 1, 0, 1, 1, 1],
    'Pclass': [3, 1, 3, 1, 2, 2, 3, 1]
}

print("Name:Suresh Dewali")
print("Roll no:1323575")

df = pd.DataFrame(data)
print("\n✅ Dataset Loaded Successfully!\n")
print(df)

# 2️⃣ Summary statistics
print("\nSummary Statistics:\n", df.describe())

# 3️⃣ Correlation analysis
print("\nCorrelation Matrix:\n", df.corr(numeric_only=True))

# 4️⃣ Analyze categorical variables
print("\nValue Counts for 'Survived':\n", df['Survived'].value_counts())
print("\nValue Counts for 'Sex':\n", df['Sex'].value_counts())
print("\nValue Counts for 'Pclass':\n", df['Pclass'].value_counts())

# 5️⃣ Create pivot tables
pivot_age = df.pivot_table(values='Age', index='Sex', columns='Survived', aggfunc='mean')
pivot_fare = df.pivot_table(values='Fare', index='Pclass', columns='Survived', aggfunc='mean')
print("\nPivot Table (Mean Age by Gender and Survival):\n", pivot_age)
print("\nPivot Table (Mean Fare by Pclass and Survival):\n", pivot_fare)

# 6️⃣ Filter data based on conditions
high_fare = df[df['Fare'] > 50]
print("\nPassengers with Fare > 50:\n", high_fare[['Name', 'Fare', 'Pclass', 'Survived']])

# 7️⃣ Sort data
sorted_df = df.sort_values(by='Fare', ascending=False)
print("\nData Sorted by Fare (High to Low):\n", sorted_df)

# 8️⃣ Add insights
avg_age_survived = df[df['Survived']==1]['Age'].mean()
avg_age_not_survived = df[df['Survived']==0]['Age'].mean()
print(f"\nAverage Age of Survived Passengers: {avg_age_survived}")
print(f"Average Age of Not Survived Passengers: {avg_age_not_survived}")
