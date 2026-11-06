import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Uvozimo funkcije za podelu podataka
from data_preprocessing import split_features_and_target


def evaluate_model(data_path: str, model_path: str):
    """
    Učitava sačuvani model, vrši predikcije nad test skupom,
    računa metrike regresije (MAE, MSE, RMSE, R2) i štampa primere.
    """
    print("Loading dataset for evaluation...")
    df = pd.read_csv(data_path)

    X, y = split_features_and_target(df)

    # Koristimo istu podelu (test_size=0.2, random_state=42) da dobijemo isti test skup
    print("Creating the test split...")
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Učitavamo sačuvani model (pipeline)
    print(f"Loading trained model from {model_path}...")
    model = joblib.load(model_path)

    # Pravimo predikcije nad test skupom
    print("Making predictions...")
    y_pred = model.predict(X_test)

    # Računamo metrike regresije
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*40)
    print("REGRESSION EVALUATION METRICS:")
    print("="*40)
    print(f"MAE  (Mean Absolute Error): {mae:.2f} $")
    print(f"MSE  (Mean Squared Error):  {mse:.2f}")
    print(f"RMSE (Root Mean Sq. Error): {rmse:.2f} $")
    print(f"R²   (Coefficient of Det.): {r2:.4f}")
    print("="*40)

    # Prikaz tabele sa nekoliko primera (stvarna vs predviđena cena)
    comparison_df = pd.DataFrame({
        "Stvarna cena ($)": y_test.values[:10],
        "Predviđena cena ($)": y_pred[:10],
        "Apsolutna greška ($)": abs(y_test.values[:10] - y_pred[:10])
    }).round(2)

    print("\nPrimeri predikcija (prvih 10 automobila):")
    print(comparison_df.to_string(index=False))

    # Čuvanje metrika u DataFrame tabelu za krajnji izveštaj
    metrics_summary = pd.DataFrame({
        "metric": ["MAE", "MSE", "RMSE", "R2"],
        "value": [mae, mse, rmse, r2]
    })

    return metrics_summary


if __name__ == "__main__":
    DATA_PATH = "data/cars_cleaned_with_features.csv"
    MODEL_PATH = "models/car_price_model.joblib"

    evaluate_model(DATA_PATH, MODEL_PATH)
