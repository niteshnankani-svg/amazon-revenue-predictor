import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load CSV from same folder as this script
BASE_DIR = os.path.dirname(__file__)
csv_path = os.path.join(BASE_DIR, "amazon_sales_dataset.csv")

df = pd.read_csv(csv_path)
print("Loaded rows, cols:", df.shape)

# Features and target
features = ["price", "quantity_sold", "discount_percent", "rating", "review_count"]
X = df[features]
y = df["total_revenue"]

print("\nX sample:")
print(X.head())
print("\ny sample:")
print(y.head())

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nShapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel trained.")

# Predict + evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = model.score(X_test, y_test)

print("\nMAE:", mae)
print("R2 :", r2)

# Compare
comparison = pd.DataFrame({
    "Actual": y_test.reset_index(drop=True).head(10),
    "Predicted": pd.Series(y_pred).head(10)
})
print("\nActual vs Predicted (first 10):")
print(comparison)

# Weights
weights_df = pd.DataFrame({"feature": features, "weight": model.coef_})
print("\nWeights:")
print(weights_df)
print("Bias:", model.intercept_)

# Make y_test position-aligned
y_test_reset = y_test.reset_index(drop=True)

# Ensure y_pred is a Series with same index 0..9999
y_pred_series = pd.Series(y_pred).reset_index(drop=True)

# Now subtraction is safe (same length, same index)
errors = y_test_reset - y_pred_series

error_df = pd.DataFrame({
    "Actual": y_test_reset,
    "Predicted": y_pred_series,
    "Error": errors
})

print(error_df.head(10))
print(len(y_test), len(y_pred))
print(y_test.index[:5])