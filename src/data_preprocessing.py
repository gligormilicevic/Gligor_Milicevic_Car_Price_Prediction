import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Definišemo ciljnu promenljivu
TARGET_COLUMN = "priceUSD"


def split_features_and_target(df: pd.DataFrame):
    """
    Razdvaja dataframe na ulazne karakteristike (X) i ciljnu promenljivu (y).
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Ciljna kolona '{TARGET_COLUMN}❜ ne postoji u datasetu!")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Kreira i vraća ColumnTransformer koji obrađuje numeričke i kategorijske kolone.
    """
    # Automatski izdvajamo numeričke i kategorijske kolone iz X
    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]).columns.tolist()

    print(f"Numeričke kolone za obradu: {numeric_features}")
    print(f"Kategorijske kolone za obradu: {categorical_features}")

    # Pipeline za numeričke podatke: popuni praznine medijanom + skaliraj
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Pipeline za kategorijske podatke: popuni praznine najčešćom vrednošću + OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Spajamo sve u jedan ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


if __name__ == "__main__":
    # Testiranje modula
    DATA_PATH = "data/cars_cleaned_with_features.csv"
    df = pd.read_csv(DATA_PATH)
    X, y = split_features_and_target(df)
    preprocessor = build_preprocessor(X)
    print("Pretprocesor je uspešno kreiran!")
