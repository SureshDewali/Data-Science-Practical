import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create a sample dataset
data = pd.DataFrame({
    'Age': np.random.randint(20, 50, 50),
    'Salary': np.random.randint(30000, 80000, 50),
    'Experience': np.random.randint(1, 15, 50),
    'Department': np.random.choice(['HR', 'IT', 'Finance', 'Marketing'], 50)
})

print("Name: Suresh Dewali")
print("Roll No: 1323575")
print("\nSample Data:\n", data.head())

# Pairplot
sns.pairplot(data, diag_kind='kde')
plt.suptitle("Pairplot of Dataset", y=1.02)
plt.show()

# Boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(x='Department', y='Salary', data=data, palette='Set2')
plt.title("Salary Distribution by Department")
plt.show()

# Correlation Heatmap
corr = data.corr(numeric_only=True)
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Multiple Seaborn Plots in One Figure
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(data['Age'], bins=10, ax=axes[0], color='skyblue')
axes[0].set_title("Age Distribution")

sns.scatterplot(x='Experience', y='Salary', data=data, ax=axes[1], color='red')
axes[1].set_title("Experience vs Salary")
plt.tight_layout()
plt.show()
