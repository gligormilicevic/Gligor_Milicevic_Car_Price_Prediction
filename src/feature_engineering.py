import pandas as pd


def engineer_features(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Dodaje nove karakteristike (features) na očišćeni skup podataka o automobilima.
    """
    print("Loading cleaned dataset...")
    df = pd.read_csv(input_path)

    print("Engineering new features...")

    # 1. Starost automobila (trenutna godina je 2026)
    CURRENT_YEAR = 2026
    df["car_age"] = CURRENT_YEAR - df["year"]

    # Da izbegnemo deljenje sa nulom ako je automobil kupljen ove godine (car_age = 0)
    # Dodajemo +1 u imeniocu da logika bude stabilna
    df["mileage_per_year"] = df["mileage_km"] / (df["car_age"] + 1)

    # 2. Zapremina motora u litrima (umesto cm3, npr. 1600 cm3 -> 1.6 L)
    # Proveravamo da li kolona volume_cm3 postoji i da li ima vrednosti
    if "volume_cm3" in df.columns:
        df["engine_volume_liters"] = df["volume_cm3"] / 1000.0

    # 3. Binarni indikatori (True/False pretvoreni u 1/0)
    # Da li je automobil noviji (star do 5 godina, proizveden od 2021. naviše)
    df["is_newer_car"] = (df["car_age"] <= 5).astype(int)

    # Da li automobil ima veoma veliku kilometražu (preko 200.000 km)
    df["is_high_mileage"] = (df["mileage_km"] > 200000).astype(int)

    print(f"Dataset shape after feature engineering: {df.shape}")

    # Čuvanje skupa sa novim karakteristikama
    df.to_csv(output_path, index=False)
    print(f"Dataset with features saved successfully to {output_path}")

    return df


if __name__ == "__main__":
    INPUT_FILE = "data/cars_cleaned.csv"
    OUTPUT_FILE = "data/cars_cleaned_with_features.csv"

    engineer_features(INPUT_FILE, OUTPUT_FILE)
