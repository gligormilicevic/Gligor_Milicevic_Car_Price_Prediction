import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import build_preprocessor, split_features_and_target


def compare_models(data_path: str, best_model_output_path: str):
    """
    Upoređuje više regresionih algoritama (Linear Regression, Decision Tree, Random Forest),
    prikazuje tabelu sa metrikama i čuva najbolji model.
    """
    print("Loading dataset for model comparison...")
    df = pd.read_csv(data_path)

    X, y = split_features_and_target(df)

    # Delimo podatke na train/test skup
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Kreiramo preprocesor
    preprocessor = build_preprocessor(X_train)

    # Definišemo rečnik modela koje želimo da testiramo
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    }

    results = []
    trained_pipelines = {}

    print("\nTraining and evaluating models...")
    print("-" * 65)
    print(f"{'Model Name':<20} | {'MAE ($)':<12} | {'RMSE ($)':<12} | {'R² Score':<10}")
    print("-" * 65)

    for name, model in models.items():
        # Kreiramo pipeline za svaki model
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

        # Treniramo model
        pipeline.fit(X_train, y_train)

        # Predikcije
        y_pred = pipeline.predict(X_test)

        # Metrike
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Pipeline": pipeline
        })

        print(f"{name:<20} | {mae:<12.2f} | {rmse:<12.2f} | {r2:<10.4f}")

    print("-" * 65)

    # Pronalazimo najbolji model na osnovu R² metrike (ili najmanjeg MAE/RMSE)
    best_result = max(results, key=lambda x: x["R2"])
    best_model_name = best_result["Model"]
    best_pipeline = best_result["Pipeline"]

    print(f"\nNajbolji model po R² rezultatu je: **{best_model_name}**!")

    # Čuvamo najbolji model u models/ folder
    os.makedirs(os.path.dirname(best_model_output_path), exist_ok=True)
    joblib.dump(best_pipeline, best_model_output_path)
    print(f"Najbolji model je uspešno sačuvan u {best_model_output_path}")


if __name__ == "__main__":
    DATA_PATH = "data/cars_cleaned_with_features.csv"
    BEST_MODEL_PATH = "models/car_price_model.joblib"

    compare_models(DATA_PATH, BEST_MODEL_PATH)
