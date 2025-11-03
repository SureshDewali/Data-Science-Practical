import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

print("Name: Suresh Dewali")
print("Roll No: 1323575\n")

# -------------------------------------------------------
# Step 1: Load Dataset (Iris dataset for classification)
# -------------------------------------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("📘 Dataset Loaded Successfully!")
print("Shape of data:", X.shape, "\n")

# -------------------------------------------------------
# Step 2: Split the dataset and standardize features
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------
# Step 3: Evaluate models using K-Fold Cross Validation
# -------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN Classifier": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

print("🔍 Model Evaluation using K-Fold Cross Validation:\n")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='accuracy')
    cv_results[name] = scores.mean()
    print(f"{name} - Average Accuracy: {scores.mean():.4f}")

# -------------------------------------------------------
# Step 4: Hyperparameter Tuning using GridSearchCV
# -------------------------------------------------------

# Tuning KNN
print("\n⚙️ Hyperparameter Tuning for KNN Classifier:")
knn_params = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='accuracy')
knn_grid.fit(X_train_scaled, y_train)
print("Best KNN Parameters:", knn_grid.best_params_)
print("Best KNN Accuracy:", knn_grid.best_score_)

# Tuning Decision Tree
print("\n🌳 Hyperparameter Tuning for Decision Tree Classifier:")
tree_params = {
    'max_depth': [2, 3, 4, 5, None],
    'criterion': ['gini', 'entropy']
}
tree_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), tree_params, cv=5, scoring='accuracy')
tree_grid.fit(X_train_scaled, y_train)
print("Best Decision Tree Parameters:", tree_grid.best_params_)
print("Best Decision Tree Accuracy:", tree_grid.best_score_)

# -------------------------------------------------------
# Step 5: Evaluate the Best Models on Test Data
# -------------------------------------------------------
best_knn = knn_grid.best_estimator_
best_tree = tree_grid.best_estimator_

best_knn.fit(X_train_scaled, y_train)
best_tree.fit(X_train_scaled, y_train)

y_pred_knn = best_knn.predict(X_test_scaled)
y_pred_tree = best_tree.predict(X_test_scaled)

print("\n✅ Final Evaluation on Test Set:\n")

print("KNN Classifier Report:\n", classification_report(y_test, y_pred_knn))
print("Decision Tree Classifier Report:\n", classification_report(y_test, y_pred_tree))

# Compare accuracies
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_tree = accuracy_score(y_test, y_pred_tree)

print(f"KNN Test Accuracy: {acc_knn:.4f}")
print(f"Decision Tree Test Accuracy: {acc_tree:.4f}")

# -------------------------------------------------------
# Step 6: Model Comparison Summary
# -------------------------------------------------------
summary = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN (Tuned)', 'Decision Tree (Tuned)'],
    'Cross-Validation Accuracy': [cv_results['Logistic Regression'], knn_grid.best_score_, tree_grid.best_score_],
    'Test Accuracy': [np.nan, acc_knn, acc_tree]
})

print("\n📊 Model Performance Summary:\n", summary)
print("\n🎯 Outcome: Students evaluated models using K-Fold Cross Validation and optimized hyperparameters using GridSearchCV successfully.")
