import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Name: Suresh Dewali")
print("Roll No: 1323575\n")

# ---------------------------------
# Step 1: Load or Create the Dataset
# ---------------------------------
# Example: Study hours vs student scores
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Scores': [35, 40, 50, 55, 65, 70, 75, 85, 90, 95]
}

df = pd.DataFrame(data)
print("📘 Original Dataset:\n", df, "\n")

# Split features and target
X = df[['Hours_Studied']]
y = df['Scores']

# ----------------------------
# Step 2: Linear Regression Model
# ----------------------------
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Predictions
y_pred_linear = lin_reg.predict(X)

# Metrics
mae = mean_absolute_error(y, y_pred_linear)
mse = mean_squared_error(y, y_pred_linear)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred_linear)

print("✅ Linear Regression Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}\n")

# Visualization: Linear Regression Line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred_linear, color='red', label='Linear Regression Line')
plt.title("Linear Regression: Study Hours vs Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# Step 3: Polynomial Regression Model
# ----------------------------
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# Metrics for Polynomial Regression
mae_poly = mean_absolute_error(y, y_pred_poly)
mse_poly = mean_squared_error(y, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(y, y_pred_poly)

print("✅ Polynomial Regression Metrics:")
print(f"Mean Absolute Error (MAE): {mae_poly:.2f}")
print(f"Mean Squared Error (MSE): {mse_poly:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_poly:.2f}")
print(f"R² Score: {r2_poly:.4f}\n")

# Visualization: Polynomial Regression Curve
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred_poly, color='green', label='Polynomial Regression (degree=2)')
plt.title("Polynomial Regression: Study Hours vs Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# Step 4: Residual Error Visualization
# ----------------------------
plt.scatter(y_pred_linear, y_pred_linear - y, color='red', label='Linear Residuals')
plt.scatter(y_pred_poly, y_pred_poly - y, color='green', label='Polynomial Residuals')
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Residual Errors Comparison")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Predicted - Actual)")
plt.legend()
plt.grid(True)
plt.show()

print("🎯 Outcome: Linear and Polynomial Regression models were implemented, evaluated, and visualized successfully.")
