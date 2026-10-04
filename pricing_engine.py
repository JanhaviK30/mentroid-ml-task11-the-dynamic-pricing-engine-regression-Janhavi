"""Dynamic Pricing Engine (Regression).

Predicts the optimal Sale_Price from Competitor_Price, Inventory_Level and
Demand_Score using Linear Regression, and plots actual vs. predicted prices.
"""
import matplotlib

matplotlib.use("Agg")  # save to file without needing a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# X columns: Competitor_Price, Inventory_Level, Demand_Score; y: Sale_Price
X, y = make_regression(n_samples=1000, n_features=3, noise=15, random_state=42)
features = ["Competitor_Price", "Inventory_Level", "Demand_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
for name, coef in zip(features, model.coef_):
    print(f"  {name} coefficient: {coef:.4f}")

# Actual vs. predicted scatter plot
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred, alpha=0.6, edgecolor="k", linewidth=0.3, label="Test samples")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "r--", label="Perfect prediction")
ax.set_xlabel("Actual Prices")
ax.set_ylabel("Predicted Prices")
ax.set_title(f"Actual vs. Predicted Prices (MAE = {mae:.2f})")
ax.legend()
fig.tight_layout()
fig.savefig("actual_vs_predicted.png", dpi=150)
print("Saved plot to actual_vs_predicted.png")
