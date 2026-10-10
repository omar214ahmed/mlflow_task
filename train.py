import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
N_ESTIMATORS = 100

CONFIGS = [
    {"run_name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
    {"run_name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
    {"run_name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
]

# Part 1: load + split (80/20, fixed seed)
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# MLflow setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("California-Housing-Price-Prediction")

# Parts 2 and 3: train, track, log
for cfg in CONFIGS:
    with mlflow.start_run(run_name=cfg["run_name"]):
        model = GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))
        r2 = float(r2_score(y_val, preds))

        mlflow.log_param("max_depth", cfg["max_depth"])
        mlflow.log_param("learning_rate", cfg["learning_rate"])
        mlflow.log_param("n_estimators", N_ESTIMATORS)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Trust the sklearn Tree type so skops can serialize the model
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(
            f"{cfg['run_name']}: max_depth={cfg['max_depth']}, "
            f"lr={cfg['learning_rate']} -> RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}"
        )

# Part 4: pick best run by RMSE (lowest wins)
runs = mlflow.search_runs(order_by=["metrics.rmse ASC"])
best = runs.iloc[0]
print("\nBest model (lowest RMSE):")
print(f"  Run name:      {best['tags.mlflow.runName']}")
print(f"  max_depth:     {best['params.max_depth']}")
print(f"  learning_rate: {best['params.learning_rate']}")
print(f"  RMSE:          {best['metrics.rmse']:.4f}")
print(f"  MAE:           {best['metrics.mae']:.4f}")
print(f"  R2:            {best['metrics.r2']:.4f}")