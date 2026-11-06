# Car Price Prediction - Machine Learning Regression Project

Ovaj projekat predstavlja kompletan regresioni mašinski radni tok (workflow) za predviđanje tržišne cene polovnih automobila na osnovu njihovih karakteristika (marka, model, godina, kilometraža, zapremina motora, tip goriva, itd.). Projekat je razvijen kao praktičan zadatak za mašinsko učenje, sa fokusom na modularnu strukturu koda, inženjering karakteristika i poređenje više algoritama.

**Autor:** Gligor Milićević

---

## 📁 Struktura projekta

car-price-prediction/
├── data/
│   ├── cars.csv                  # Sirovi skup podataka
│   ├── cars_cleaned.csv          # Očišćeni podaci
│   └── cars_cleaned_with_features.csv # Podaci sa dodatim karakteristikama
├── notebooks/
│   └── 01_eda.ipynb             # Jupyter sveska sa eksploratornom analizom
├── src/
│   ├── data_cleaning.py         # Skripta za čišćenje i korekciju podataka
│   ├── feature_engineering.py   # Skripta za dodavanje novih karakteristika
│   ├── data_preprocessing.py    # Skripta sa ColumnTransformer-om i pipeline-om
│   ├── model_training.py        # Skripta za treniranje osnovnog modela
│   ├── model_evaluation.py      # Skripta za izračunavanje MAE, MSE, RMSE, R²
│   └── model_comparison.py      # Skripta za poređenje više algoritama
├── models/
│   └── car_price_model.joblib   # Sačuvan finalni model (Random Forest)
├── README.md                    # Dokumentacija projekta
└── requirements.txt             # Potrebne Python biblioteke