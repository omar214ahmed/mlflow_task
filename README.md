# MLflow Experiment Tracking: House Price Prediction

## 1. Task Description

This project uses MLflow to track and compare machine learning experiments on a
House Price Prediction problem. The same regression model is trained three times
with different hyperparameters. For each run, MLflow records the parameters,
validation metrics, and the trained model as an artifact. The best model is then
selected using RMSE as the primary metric.

- **Dataset:** California Housing (`sklearn.datasets.fetch_california_housing`)
- **Split:** 80% training / 20% validation (`random_state=42`)
- **Model:** `GradientBoostingRegressor` (`n_estimators=100`, `random_state=42`)
- **Tracked parameters:** `max_depth`, `learning_rate`
- **Tracked metrics:** RMSE, MAE, R²

## 2. How to Run

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the three experiments
python train.py

# 4. Open the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000 in your browser and select the experiment
**California-Housing-Price-Prediction**.

## 3. MLflow Experiment Results

![MLflow experiment results](screenshots/mlflow_results.png)

## 4. Comparison Table

| Run   | Max Depth | Learning Rate | RMSE   | MAE    | R²     |
|-------|-----------|---------------|--------|--------|--------|
| Run 1 | 3         | 0.1           | 0.5422 | 0.3716 | 0.7756 |
| Run 2 | 5         | 0.05          | **0.5198** | **0.3531** | **0.7938** |
| Run 3 | 7         | 0.01          | 0.6973 | 0.5311 | 0.6290 |

## 5. Selected Best Model

**Run 2**: `max_depth = 5`, `learning_rate = 0.05`
(RMSE = 0.5198, MAE = 0.3531, R² = 0.7938)

## 6. Why I Selected It

RMSE was the primary selection metric, and Run 2 has the lowest validation RMSE
(0.5198), compared with 0.5422 for Run 1 and 0.6973 for Run 3. Run 2 also has
the best MAE and the highest R², so it is the best model on all three metrics.

Run 3 performed worst. With a learning rate of only 0.01 and a fixed number of
100 trees, the model learns too slowly and underfits, even though its trees are
the deepest. Run 1 has shallow trees (depth 3), which limits how much it can
capture. Run 2 balances tree depth and learning rate best for this setup.
