import pandas as pd

def clean_car_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Učitava sirove podatke o automobilima, čisti nazive kolona, 
    uklanja duplikate, rešava nedostajuće vrednosti i čisti ekstreme.
    """
    print("Loading raw dataset...")
    df = pd.read_csv(input_path)
    
    print(f"Original shape: {df.shape}")
    
    # 1. Preimenovanje kolona da uklonimo zagrade i specijalne karaktere
    df = df.rename(columns={
        "mileage(kilometers)": "mileage_km",
        "volume(cm3)": "volume_cm3"
    })
    
    # 2. Uklanjanje duplikata ako postoje
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df)} duplicate rows.")
    
    # 3. Uklanjanje redova gde nedostaju ključne vrednosti (Cena, Godina, Kilometraža)
    critical_columns = ["priceUSD", "year", "mileage_km", "make", "model"]
    df = df.dropna(subset=critical_columns)
    
    # 4. Čišćenje nerealnih vrednosti (ekstrema)
    # Cena mora biti veća od 500$ (da izbacimo nerealne poklone ili greške)
    df = df[(df["priceUSD"] > 500) & (df["priceUSD"] < 200000)]
    
    # Godina proizvodnje mora biti između 1960. i tekuće godine (2026)
    df = df[(df["year"] >= 1960) & (df["year"] <= 2026)]
    
    # Kilometraža ne može biti negativna niti preko 1.200.000 km
    df = df[(df["mileage_km"] >= 0) & (df["mileage_km"] <= 1200000)]
    
    print(f"Cleaned shape: {df.shape}")
    
    # 5. Čuvanje očišćenih podataka u novi fajl (opciono, ili ih vraćamo za dalju obradu)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved successfully to {output_path}")
    
    return df

if __name__ == "__main__":
    INPUT_FILE = "data/cars.csv"
    OUTPUT_FILE = "data/cars_cleaned.csv"
    
    clean_car_data(INPUT_FILE, OUTPUT_FILE)