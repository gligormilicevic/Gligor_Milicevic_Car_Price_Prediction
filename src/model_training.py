import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Uvozimo funkcije iz našeg prethodnog modula za pretprocesiranje
from data_preprocessing import build_preprocessor, split_features_and_target


def train_and_save_model(data_path: str, model_output_path: str):
    """
    Učitava podatke, deli ih na train/test, kreira pipeline (preprocesor + model),
    trenira model i čuva ga na disku.
    """
    print("Loading dataset with features...")
    df = pd.read_csv(data_path)

    # 1. Razdvajamo X (karakteristike) i y (ciljna promenljiva - cena)
    X, y = split_features_and_target(df)

    # 2. Delimo podatke na trening i test skup (80% trening, 20% test, random_state=42)
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # 3. Kreiramo pretprocesor na osnovu trening podataka
    preprocessor = build_preprocessor(X_train)

    # 4. Kreiramo celac Pipeline: Pretprocesiranje + Linearna Regresija
    print("Building pipeline with Linear Regression...")
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # 5. Treniramo model
    print("Training the model...")
    model_pipeline.fit(X_train, y_train)
    print("Model training completed successfully!")

    # 6. Čuvamo model u folder models/
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model_pipeline, model_output_path)
    print(f"Trained model saved successfully to {model_output_path}")


if __name__ == "__main__":
    DATA_PATH = "data/cars_cleaned_with_features.csv"
    MODEL_OUTPUT = "models/car_price_model.joblib"

    train_and_save_model(DATA_PATH, MODEL_OUTPUT)
