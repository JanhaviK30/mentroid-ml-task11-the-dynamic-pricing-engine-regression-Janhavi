# Dynamic Pricing Engine (Regression)

Predicts the optimal `Sale_Price` from `Competitor_Price`, `Inventory_Level` and `Demand_Score`
using Linear Regression (80/20 train/test split) on `make_regression` data.

Run: `pip install -r requirements.txt && python pricing_engine.py`

Outputs the Mean Absolute Error and saves `actual_vs_predicted.png`.
