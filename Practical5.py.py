import matplotlib.pyplot as plt
import numpy as np

# Data
x = np.arange(1, 11)
y1 = x * 2
y2 = x ** 2
y3 = np.random.randint(10, 100, 10)

print("Name:Suresh Dewali")
print("Roll no:1323575")
# Line Plot
plt.figure(figsize=(8, 5))
plt.plot(x, y1, label='y = 2x', color='blue', marker='o')
plt.plot(x, y2, label='y = x²', color='red', linestyle='--')
plt.title("Line Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.show()

# Bar Plot
plt.figure(figsize=(6, 4))
plt.bar(x, y3, color='orange')
plt.title("Bar Plot Example")
plt.xlabel("X values")
plt.ylabel("Random Values")
plt.show()

# Histogram
data = np.random.randn(1000)
plt.figure(figsize=(6, 4))
plt.hist(data, bins=20, color='green', alpha=0.7)
plt.title("Histogram Example")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot
plt.figure(figsize=(6, 4))
plt.scatter(x, y3, color='purple', marker='*')
plt.title("Scatter Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# Subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].plot(x, y1, color='blue'); axs[0, 0].set_title("Line Plot")
axs[0, 1].bar(x, y3, color='orange'); axs[0, 1].set_title("Bar Plot")
axs[1, 0].hist(data, bins=15, color='green'); axs[1, 0].set_title("Histogram")
axs[1, 1].scatter(x, y3, color='purple'); axs[1, 1].set_title("Scatter Plot")
plt.tight_layout()
plt.show()
